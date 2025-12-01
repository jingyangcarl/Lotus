import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from math import log10

# Optional imports for cross-verification
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

try:
    from torchvision.models import vgg16, VGG16_Weights
    from torchvision.transforms import Normalize
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x



def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b

def pad_input(img, intrins, lrtb=(0,0,0,0)):
    """ pad input image
        img should be a torch tensor of shape (B, 3, H, W)
        intrins should be a torch tensor of shape (B, 3, 3)
    """
    l, r, t, b = lrtb
    if l+r+t+b != 0:
        # pad_value_R = (0 - 0.485) / 0.229
        # pad_value_G = (0 - 0.456) / 0.224
        # pad_value_B = (0 - 0.406) / 0.225
        pad_value_R = 0
        pad_value_G = 0
        pad_value_B = 0

        img_R = F.pad(img[:,0:1,:,:], (l, r, t, b), mode="constant", value=pad_value_R)
        img_G = F.pad(img[:,1:2,:,:], (l, r, t, b), mode="constant", value=pad_value_G)
        img_B = F.pad(img[:,2:3,:,:], (l, r, t, b), mode="constant", value=pad_value_B)

        img = torch.cat([img_R, img_G, img_B], dim=1)

        if intrins is not None:
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
    return img, intrins

def compute_cosine_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error

def compute_normal_metrics(total_normal_errors):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics


def mse(pred, gt, mask=None, reduction='mean'):
    """Mean squared error between pred and gt.

    pred, gt: torch.Tensor of shape (B, C, H, W) or (C, H, W)
    mask: optional tensor broadcastable to pred, with 1 for valid pixels
    reduction: 'mean' (scalar), 'none' (per-image tensor)
    """
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    err = (pred - gt) ** 2
    if mask is not None:
        mask = _to_tensor(mask).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        # expand mask to same shape
        mask = mask.expand_as(err)
        err = err * mask
        denom = mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
        per_image = err.reshape(err.size(0), -1).sum(dim=1) / denom
    else:
        per_image = err.reshape(err.size(0), -1).mean(dim=1)

    if reduction == 'mean':
        return per_image.mean()
    elif reduction == 'none':
        return per_image
    else:
        raise ValueError('unknown reduction')


def psnr(pred, gt, data_range=1.0, mask=None):
    """Peak signal-to-noise ratio in dB.

    data_range: maximum possible pixel value (e.g., 1.0 for normalized images)
    """
    per_image_mse = mse(pred, gt, mask=mask, reduction='none')
    # avoid division by zero
    eps = 1e-10
    psnr_per = 10.0 * torch.log10((data_range ** 2) / (per_image_mse + eps))
    return psnr_per.mean()


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, gt, window_size=11, data_range=1.0, K=(0.01, 0.03)):
    """Compute single-scale SSIM for batches of images.

    pred, gt: torch.Tensor (B, C, H, W) in [0, data_range]
    returns mean SSIM over the batch
    """
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    _, channel, _, _ = pred.size()
    window = _create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, groups=channel, padding=window_size // 2)
    mu2 = F.conv2d(gt, window, groups=channel, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, groups=channel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, groups=channel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, groups=channel, padding=window_size // 2) - mu1_mu2

    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # average over channels and spatial dims
    return ssim_map.mean()


def lpips(pred, gt, net='vgg'):
    """Compute LPIPS perceptual distance.

    If the `lpips` package is installed, it will be used. Otherwise a lightweight
    VGG-based feature L2 distance is computed as a fallback.
    pred, gt: torch.Tensor in [0,1], shape (B,3,H,W)
    """
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    # try to use lpips package if available
    try:
        import lpips as _lpips

        model = _lpips.LPIPS(net=net)
        # LPIPS expects inputs in [-1,1]
        inp = pred * 2.0 - 1.0
        tgt = gt * 2.0 - 1.0
        with torch.no_grad():
            val = model.forward(inp, tgt)
        # lpips returns (B,1,1,1) or (B,1)
        return val.view(-1).mean()
    except Exception:
        # fallback: VGG-based feature distance
        try:
            from torchvision import models
            from torchvision.models.feature_extraction import create_feature_extractor
            from torchvision.transforms import Normalize

            device = pred.device
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
            return _vgg_perceptual_distance(pred, gt, vgg)
        except Exception:
            raise RuntimeError('lpips is not available and torchvision fallback failed; install "lpips" or ensure torchvision pretrained weights are available')


def _vgg_perceptual_distance(pred, gt, vgg_model):
    """Simple VGG feature L2 distance fallback for LPIPS.

    Extracts features at several layers and averages normalized L2 distances.
    """
    from torchvision.models.feature_extraction import create_feature_extractor
    from torchvision.transforms import Normalize

    device = pred.device
    # normalize input to ImageNet stats
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    B = pred.shape[0]
    # apply normalization per image
    inp = pred.clone()
    tgt = gt.clone()
    for i in range(B):
        inp[i] = normalize(inp[i])
        tgt[i] = normalize(tgt[i])

    return_nodes = {
        'features.3': 'f1',  # relu1_2
        'features.8': 'f2',  # relu2_2
        'features.15': 'f3', # relu3_3
        'features.22': 'f4'  # relu4_3
    }
    feat_extractor = create_feature_extractor(vgg_model, return_nodes=return_nodes).to(device).eval()
    with torch.no_grad():
        feats_pred = feat_extractor(inp)
        feats_gt = feat_extractor(tgt)

    # compute normalized L2 per layer
    layer_weights = [1.0 / len(return_nodes)] * len(return_nodes)
    total = 0.0
    for i, key in enumerate(sorted(return_nodes.keys(), key=lambda x: int(x.split('.')[1]))):
        f_pred = feats_pred[return_nodes[key]]
        f_gt = feats_gt[return_nodes[key]]
        # normalize channels
        # shape B x C x H x W
        N = f_pred.size(1)
        # L2 norm across channels
        diff = (f_pred - f_gt)
        # spatial average and channel average
        d = diff.pow(2).mean(dim=[1,2,3])
        total = total + layer_weights[i] * d

    # return mean across batch
    return total.mean()


# Cross-verification functions using torchmetrics and torchvision
def mse_torchmetrics(pred, gt, mask=None):
    """MSE using torchmetrics for cross-verification."""
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics not available")
    
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    device = pred.device
    
    # Flatten for torchmetrics
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    
    if mask is not None:
        mask = _to_tensor(mask).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask_flat = mask.reshape(-1)
        # Only use valid pixels
        valid_idx = mask_flat > 0
        pred_flat = pred_flat[valid_idx]
        gt_flat = gt_flat[valid_idx]
    
    metric = torchmetrics.MeanSquaredError().to(device)
    return metric(pred_flat, gt_flat)


def psnr_torchmetrics(pred, gt, data_range=1.0, mask=None):
    """PSNR using torchmetrics for cross-verification."""
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics not available")
    
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    device = pred.device
    
    if mask is not None:
        # Apply mask
        mask = _to_tensor(mask).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(pred)
        pred = pred * mask
        gt = gt * mask
    
    metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range).to(device)
    return metric(pred, gt)


def ssim_torchmetrics(pred, gt, data_range=1.0):
    """SSIM using torchmetrics for cross-verification."""
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics not available")
    
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    device = pred.device
    
    metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    return metric(pred, gt)


def lpips_torchmetrics(pred, gt):
    """LPIPS using torchmetrics for cross-verification."""
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics not available")
    
    pred = _to_tensor(pred).float()
    gt = _to_tensor(gt).float()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    device = pred.device
    
    # torchmetrics LPIPS expects input in [0,1]
    metric = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    return metric(pred, gt)


def cross_verify_metrics(pred, gt, data_range=1.0, mask=None, tolerance=1e-3):
    """Cross-verify custom implementations against torchmetrics.
    
    Returns a dict with comparison results and relative differences.
    """
    results = {}
    
    # MSE comparison
    try:
        custom_mse = mse(pred, gt, mask=mask, reduction='mean')
        if TORCHMETRICS_AVAILABLE:
            tm_mse = mse_torchmetrics(pred, gt, mask=mask)
            rel_diff = abs(custom_mse - tm_mse) / (tm_mse + 1e-8)
            results['mse'] = {
                'custom': float(custom_mse),
                'torchmetrics': float(tm_mse),
                'rel_diff': float(rel_diff),
                'match': rel_diff < tolerance
            }
        else:
            results['mse'] = {'custom': float(custom_mse), 'torchmetrics': 'N/A'}
    except Exception as e:
        results['mse'] = {'error': str(e)}
    
    # PSNR comparison
    try:
        custom_psnr = psnr(pred, gt, data_range=data_range, mask=mask)
        if TORCHMETRICS_AVAILABLE:
            tm_psnr = psnr_torchmetrics(pred, gt, data_range=data_range, mask=mask)
            rel_diff = abs(custom_psnr - tm_psnr) / (tm_psnr + 1e-8)
            results['psnr'] = {
                'custom': float(custom_psnr),
                'torchmetrics': float(tm_psnr),
                'rel_diff': float(rel_diff),
                'match': rel_diff < tolerance
            }
        else:
            results['psnr'] = {'custom': float(custom_psnr), 'torchmetrics': 'N/A'}
    except Exception as e:
        results['psnr'] = {'error': str(e)}
    
    # SSIM comparison
    try:
        custom_ssim = ssim(pred, gt, data_range=data_range)
        if TORCHMETRICS_AVAILABLE:
            tm_ssim = ssim_torchmetrics(pred, gt, data_range=data_range)
            rel_diff = abs(custom_ssim - tm_ssim) / (tm_ssim + 1e-8)
            results['ssim'] = {
                'custom': float(custom_ssim),
                'torchmetrics': float(tm_ssim),
                'rel_diff': float(rel_diff),
                'match': rel_diff < tolerance
            }
        else:
            results['ssim'] = {'custom': float(custom_ssim), 'torchmetrics': 'N/A'}
    except Exception as e:
        results['ssim'] = {'error': str(e)}
    
    # LPIPS comparison (only if both have 3 channels)
    try:
        if pred.shape[-3] == 3:  # RGB channels
            custom_lpips = lpips(pred, gt)
            if TORCHMETRICS_AVAILABLE:
                tm_lpips = lpips_torchmetrics(pred, gt)
                rel_diff = abs(custom_lpips - tm_lpips) / (tm_lpips + 1e-8)
                results['lpips'] = {
                    'custom': float(custom_lpips),
                    'torchmetrics': float(tm_lpips),
                    'rel_diff': float(rel_diff),
                    'match': rel_diff < tolerance
                }
            else:
                results['lpips'] = {'custom': float(custom_lpips), 'torchmetrics': 'N/A'}
        else:
            results['lpips'] = {'error': 'LPIPS requires 3-channel input'}
    except Exception as e:
        results['lpips'] = {'error': str(e)}
    
    return results


def demo_cross_verification():
    """Demo function to test cross-verification with synthetic data."""
    # Create synthetic test data
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    pred = torch.rand(B, C, H, W)
    gt = torch.rand(B, C, H, W)
    
    # Optional mask (valid pixels)
    mask = torch.ones(B, 1, H, W)
    mask[:, :, :16, :16] = 0  # mask out top-left corner
    
    print("Cross-verification results:")
    print("=" * 50)
    
    results = cross_verify_metrics(pred, gt, data_range=1.0, mask=mask)
    
    for metric_name, result in results.items():
        print(f"\n{metric_name.upper()}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Custom: {result['custom']:.6f}")
            if 'torchmetrics' in result and result['torchmetrics'] != 'N/A':
                print(f"  TorchMetrics: {result['torchmetrics']:.6f}")
                print(f"  Relative Diff: {result['rel_diff']:.2e}")
                print(f"  Match: {result['match']}")
            else:
                print(f"  TorchMetrics: {result.get('torchmetrics', 'N/A')}")
    
    return results