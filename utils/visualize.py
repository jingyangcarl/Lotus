import cv2
import numpy as np
import os

import torch

from matplotlib import cm
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('root')



def tensor_to_numpy(tensor_in):
    """ torch tensor to numpy array
    """
    if tensor_in is not None:
        if tensor_in.ndim == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif tensor_in.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception('invalid tensor size')
    return tensor_in

# def unnormalize(img_in, img_stats={'mean': [0.485, 0.456, 0.406], 
#                                     'std': [0.229, 0.224, 0.225]}):
def unnormalize(img_in, img_stats={'mean': [0.5,0.5,0.5], 'std': [0.5,0.5,0.5]}):
    """ unnormalize input image
    """
    if torch.is_tensor(img_in):
        img_in = tensor_to_numpy(img_in)

    img_out = np.zeros_like(img_in)
    for ich in range(3):
        img_out[..., ich] = img_in[..., ich] * img_stats['std'][ich]
        img_out[..., ich] += img_stats['mean'][ich]
    img_out = (img_out * 255.0).astype(np.uint8)
    return img_out

def normal_to_rgb(normal, normal_mask=None, mask_background_white=True):
    """ surface normal map to RGB
        (used for visualization)

        NOTE: x, y, z are mapped to R, G, B
        NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
        normal_mask = tensor_to_numpy(normal_mask)

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    # normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    normal_rgb = ((normal + 1) * 0.5)
    if normal_mask is not None:
        if mask_background_white:
            normal_rgb = normal_rgb * normal_mask + (1.0 - normal_mask) # (B, H, W, 3)
        else:
            normal_rgb = normal_rgb * normal_mask     # (B, H, W, 3)
    return normal_rgb

def albedo_to_rgb(albedo, albedo_mask=None, mask_background_white=True, background_img=None):
    """ surface normal map to RGB
        (used for visualization)

        NOTE: x, y, z are mapped to R, G, B
        NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(albedo):
        albedo = tensor_to_numpy(albedo)
        albedo_mask = tensor_to_numpy(albedo_mask)

    # albedo_rgb = ((albedo + 1) * 0.5 * 255).astype(np.uint8)
    # albedo_rgb = ((albedo + 1) * 0.5)
    albedo_rgb = albedo
    if albedo_mask is not None:
        if mask_background_white:
            albedo_rgb = albedo_rgb * albedo_mask + (1.0 - albedo_mask) # (B, H, W, 3)
        else:
            albedo_rgb = albedo_rgb * albedo_mask     # (B, H, W, 3)
            
        if background_img is not None:
            albedo_rgb = albedo_rgb * albedo_mask + background_img * (1.0 - albedo_mask)
    return albedo_rgb

def kappa_to_alpha(pred_kappa, to_numpy=True):
    """ Confidence kappa to uncertainty alpha
        Assuming AngMF distribution (introduced in https://arxiv.org/abs/2109.09881)
    """
    if torch.is_tensor(pred_kappa) and to_numpy:
        pred_kappa = tensor_to_numpy(pred_kappa)

    if torch.is_tensor(pred_kappa):
        alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((torch.exp(- pred_kappa * np.pi) * np.pi) / (1 + torch.exp(- pred_kappa * np.pi)))
        alpha = torch.rad2deg(alpha)
    else:
        alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
                + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
        alpha = np.degrees(alpha)

    return alpha


def visualize_normal(target_dir, prefixs, img, pred_norm, pred_kappa,
                        gt_norm, gt_norm_mask, pred_error, num_vis=-1):
    """ visualize normal
    """
    error_max = 60.0

    img = tensor_to_numpy(img)                      # (B, H, W, 3)
    pred_norm = tensor_to_numpy(pred_norm)          # (B, H, W, 3)
    pred_kappa = tensor_to_numpy(pred_kappa)        # (B, H, W, 1)
    gt_norm = tensor_to_numpy(gt_norm)              # (B, H, W, 3)
    gt_norm_mask = tensor_to_numpy(gt_norm_mask)    # (B, H, W, 1)
    pred_error = tensor_to_numpy(pred_error)        # (B, H, W, 1)

    num_vis = len(prefixs) if num_vis == -1 else num_vis
    for i in range(num_vis):
        # img
        img_ = unnormalize(img[i, ...])
        target_path = '%s/%s/img.png' % (target_dir, prefixs[i])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        img = albedo_to_rgb(img[i, ...], gt_norm_mask[i, ...])
        plt.imsave(target_path, img)

        # pred_norm 
        target_path = '%s/%s/norm.png' % (target_dir, prefixs[i])
        pred_norm = normal_to_rgb(pred_norm[i, ...], gt_norm_mask[i, ...])
        plt.imsave(target_path, pred_norm)

        # pred_kappa
        if pred_kappa is not None:
            pred_alpha = kappa_to_alpha(pred_kappa[i, :, :, 0])
            target_path = '%s/%s/pred_alpha.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, pred_alpha, vmin=0.0, vmax=error_max, cmap='jet')

        # gt_norm, pred_error
        if gt_norm is not None:
            target_path = '%s/%s/gt.png' % (target_dir, prefixs[i])
            gt_norm = normal_to_rgb(gt_norm[i, ...], gt_norm_mask[i, ...])
            plt.imsave(target_path, gt_norm)

            E = pred_error[i, :, :, 0] * gt_norm_mask[i, :, :, 0]
            target_path = '%s/%s/pred_error.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, E, vmin=0, vmax=error_max, cmap='jet')
            
            # reload error image add apply normal_to_rgb to mask it out
            E_img = plt.imread(target_path)
            E_img = albedo_to_rgb(E_img, gt_norm_mask[i, ...], mask_background_white=True)
            plt.imsave(target_path, E_img)

        # img, albedo, gt, error
        all_imgs = [img, pred_norm, gt_norm, plt.imread('%s/%s/pred_error.png' % (target_dir, prefixs[i]))[...,:3]]
        all_imgs = np.concatenate(all_imgs, axis=1)  # (H, W * 4, 3)
        target_path = '%s/%s.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, all_imgs)

    return all_imgs

def visualize_albedo(target_dir, prefixs, img, pred_albedo, pred_kappa,
                        gt_albedo, gt_albedo_mask, pred_error, num_vis=-1, background_img=None):
    """ visualize albedo
    """
    error_max = 60.0

    img = tensor_to_numpy(img)                      # (B, H, W, 3)
    pred_albedo = tensor_to_numpy(pred_albedo)          # (B, H, W, 3)
    pred_kappa = tensor_to_numpy(pred_kappa)        # (B, H, W, 1)
    gt_albedo = tensor_to_numpy(gt_albedo)              # (B, H, W, 3)
    gt_albedo_mask = tensor_to_numpy(gt_albedo_mask)    # (B, H, W, 1)
    pred_error = tensor_to_numpy(pred_error)        # (B, H, W, 1)

    # check if background_img shape matches, if not, crop from the center to match aspect ratio of img and resize
    if background_img is not None:
        background_img = tensor_to_numpy(background_img)  # (B, H, W, 3)
        if background_img.shape[1:3] != img.shape[1:3]:
            ih, iw = img.shape[1:3]
            bh, bw = background_img.shape[1:3]
            scale = max(ih / bh, iw / bw)
            new_bh = int(bh * scale)
            new_bw = int(bw * scale)
            background_img_resized = cv2.resize(background_img[0], (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)[None, ...]  # (1, new_bh, new_bw, 3)
            # crop center
            start_y = (new_bh - ih) // 2
            start_x = (new_bw - iw) // 2
            background_img_cropped = background_img_resized[:, start_y:start_y+ih, start_x:start_x+iw, :]
            background_img = background_img_cropped  # (1, H, W, 3)
    else:
        background_img = torch.ones_like(img)  # white background

    num_vis = len(prefixs) if num_vis == -1 else num_vis
    for i in range(num_vis):
        # img
        img_ = unnormalize(img[i, ...])
        target_path = '%s/%s/img.png' % (target_dir, prefixs[i])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        img = albedo_to_rgb(img[i, ...], gt_albedo_mask[i, ...])
        plt.imsave(target_path, img)

        # pred_norm 
        target_path = '%s/%s/pred_albedo.png' % (target_dir, prefixs[i])
        pred_albedo = albedo_to_rgb(pred_albedo[i, ...], gt_albedo_mask[i, ...], background_img=background_img[i,...])
        plt.imsave(target_path, pred_albedo)

        # pred_kappa
        if pred_kappa is not None:
            pred_alpha = kappa_to_alpha(pred_kappa[i, :, :, 0])
            target_path = '%s/%s/pred_alpha.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, pred_alpha, vmin=0.0, vmax=error_max, cmap='jet')

        # gt_norm, pred_error
        if gt_albedo is not None:
            target_path = '%s/%s/gt_albedo.png' % (target_dir, prefixs[i])
            gt_albedo = albedo_to_rgb(gt_albedo[i, ...], gt_albedo_mask[i, ...], background_img=background_img[i,...])
            plt.imsave(target_path, gt_albedo)

            E = pred_error[i, :, :, 0] * gt_albedo_mask[i, :, :, 0]
            target_path = '%s/%s/pred_error.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, E, vmin=0, vmax=error_max, cmap='jet')
            
            # reload error image add apply normal_to_rgb to mask it out
            E_img = plt.imread(target_path)
            E_img = albedo_to_rgb(E_img, gt_albedo_mask[i, ...], mask_background_white=True)
            plt.imsave(target_path, E_img)
                
        # img, albedo, gt, error
        all_imgs = [img, pred_albedo, gt_albedo, plt.imread('%s/%s/pred_error.png' % (target_dir, prefixs[i]))[...,:3]]
        if background_img is not None:
            all_imgs.insert(0, background_img_resized[i,...])
        all_imgs = np.concatenate(all_imgs, axis=1)  # (H, W * 4, 3)
        target_path = '%s/%s.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, all_imgs)
        
    return all_imgs


def visualize_img(target_dir, prefixs, img, albedo, irradiance, pred_img, pred_kappa,
                        pred_error, num_vis=-1):
    """ visualize albedo
    """
    error_max = 60.0

    img = tensor_to_numpy(img)                      # (B, H, W, 3)
    albedo = tensor_to_numpy(albedo)                # (B, H, W, 3)
    irradiance = tensor_to_numpy(irradiance)        # (B, H, W, 3)
    pred_img = tensor_to_numpy(pred_img)            # (B, H, W, 3)
    pred_kappa = tensor_to_numpy(pred_kappa)        # (B, H, W, 1)
    pred_error = tensor_to_numpy(pred_error)        # (B, H, W, 1)

    num_vis = len(prefixs) if num_vis == -1 else num_vis
    for i in range(num_vis):
        # img
        img_ = unnormalize(img[i, ...])
        target_path = '%s/%s/img.png' % (target_dir, prefixs[i])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        plt.imsave(target_path, img[i, ...])
        
        # albedo
        target_path = '%s/%s/albedo.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, albedo[i, ...])
        
        # irradiance
        target_path = '%s/%s/irradiance.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, irradiance[i, ...])

        # pred_norm 
        target_path = '%s/%s/pred_img.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, pred_img[i, ...])

        # pred_kappa
        if pred_kappa is not None:
            pred_alpha = kappa_to_alpha(pred_kappa[i, :, :, 0])
            target_path = '%s/%s/pred_alpha.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, pred_alpha, vmin=0.0, vmax=error_max, cmap='jet')
                
        # img, albedo, gt, error
        all_imgs = [img[i,...], irradiance[i, ...], pred_img[i, ...]]
        all_imgs = np.concatenate(all_imgs, axis=1)  # (H, W * 4, 3)
        target_path = '%s/%s.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, all_imgs)
        
    return all_imgs