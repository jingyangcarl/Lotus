import sys
import os
import imageio
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

def hdr2ldr(hdr, percentile=95):
    max_val = np.percentile(hdr, percentile)
    ldr = np.clip(hdr / max_val * 255.0, 0, 255).astype(np.uint8)
    return ldr

# Apply mask and place the error map at the lower-left corner of each image
def add_error_map_with_mask(base_img, error_map, mask, scale=0.3, position="left", offset_x=0, offset_y=0):
    """
    Add a masked and resized error map to one of the bottom corners of base_img, 
    with optional pixel offsets.

    Args:
        base_img (np.ndarray): Base RGB image (H, W, 3)
        error_map (np.ndarray): Error map image (H, W, 3)
        mask (np.ndarray): Mask image (H, W, 1) or (H, W)
        scale (float): Scale of the error map relative to base image height/width
        position (str): "left" or "right" (bottom corner)
        offset_x (int): Horizontal offset in pixels (positive moves inward)
        offset_y (int): Vertical offset in pixels (positive moves upward)
    """
    H, W = base_img.shape[:2]

    # Normalize and apply mask
    mask_resized = np.clip(mask, 0, 1)
    if mask_resized.ndim == 2:
        mask_resized = mask_resized[..., None]
    mask_resized = np.repeat(mask_resized, 3, axis=-1)

    error_map = (error_map.astype(np.float32) / 255.0) * mask_resized
    error_map = (error_map * 255).astype(np.uint8)

    # Resize error map and mask
    new_h, new_w = int(H * scale), int(W * scale)
    error_map_small = cv2.resize(error_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(mask_resized, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    composed = base_img.copy()

    # Compute position
    if position.lower() == "left":
        x1 = offset_x
        x2 = x1 + new_w
    elif position.lower() == "right":
        x2 = W - offset_x
        x1 = x2 - new_w
    else:
        raise ValueError("position must be 'left' or 'right'")

    y2 = H - offset_y
    y1 = y2 - new_h

    # Ensure bounds are valid
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    # Blend error map into the selected region
    roi = composed[y1:y2, x1:x2]
    composed[y1:y2, x1:x2] = (
        error_map_small[:y2 - y1, :x2 - x1] * mask_small[:y2 - y1, :x2 - x1] +
        roi * (1 - mask_small[:y2 - y1, :x2 - x1])
    )

    return composed

def compose_lightstage_relighting():
    
    x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_posttrain_via_gt'
    x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_posttrain_via_rgb2x_posttrain'
    x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_pretrain_via_rgb2x_pretrain'
    # cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
    cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/fixed20_via1/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
    mask_root = '/home/jyang/data/LightStageObjectDB/datasets/exr/v1.3/v1.3_2/fit_512'
    
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/comparison'
    
    objs = sorted(os.listdir(x2rgb_posttrain_via_gt_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    for obj in objs:
        print(obj)
        
        if obj != 'sodacan':
            continue
        
        
        x2rgb_posttrain_via_gt = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_gbuffer', 'separated')
        light_ids = sorted(os.listdir(x2rgb_posttrain_via_gt))
        light_ids = [idx for idx in light_ids if not idx.endswith('.png')]
        
        n = 20
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        
        mosaics = []
        mosaics_selected = []
        for i, light_id in enumerate(light_ids):

            x2rgb_posttrain_via_gt_gbuffer_light_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_gt_polarization_light_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_pretrain_via_rgb2x_pretrain_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_forward_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_albedo.png')
            irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
            gt_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
            mask_path = os.path.join(mask_root, obj, 'cam07', 'mask.png')
            
            i = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            if i == 0:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{1:06d}.exr')
            else:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{2+olat_gt_map[i-1]:06d}.exr')

            x2rgb_posttrain_via_gt_gbuffer_light_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
            x2rgb_posttrain_via_gt_polarization_light_error_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_pretrain_via_rgb2x_pretrain_error_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_forward_error_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_error.png')
            
            x2rgb_posttrain_via_gt_gbuffer_light = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_path)[...,:3]
            cosmos_inverse_forward = imageio.imread(cosmos_inverse_forward_path)[...,:3]
            irradiance = imageio.imread(irradiance_path)[...,:3]
            gt = imageio.imread(gt_path)[...,:3]
            mask = imageio.imread(mask_path)[..., None] / 255.
            gt_exr = imageio.imread(gt_exr_path)[...,:3]
            gt_exr_ldr = hdr2ldr(gt_exr, 99)
            gt_exr_ldr = (gt_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)

            x2rgb_posttrain_via_gt_gbuffer_light_error = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_error_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light_error = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain_error = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_error_path)[...,:3]
            cosmos_inverse_forward_error = imageio.imread(cosmos_inverse_forward_error_path)[...,:3]

            # put the error map to the lower left corner
            # After reading all the images and errors...
            x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, x2rgb_posttrain_via_gt_gbuffer_light_error, mask, scale=0.4)
            x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, x2rgb_posttrain_via_gt_polarization_light_error, mask, scale=0.4)
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer, x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error, mask, scale=0.4)
            x2rgb_posttrain_via_rgb2x_posttrain_polarization = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_polarization, x2rgb_posttrain_via_rgb2x_posttrain_polarization_error, mask, scale=0.4)
            x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, x2rgb_pretrain_via_rgb2x_pretrain_error, mask, scale=0.4)
            cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, cosmos_inverse_forward_error, mask, scale=0.4)


            composed_img = np.concatenate([
                irradiance,
                cosmos_inverse_forward,
                x2rgb_pretrain_via_rgb2x_pretrain,
                # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer,
                # x2rgb_posttrain_via_rgb2x_posttrain_polarization,
                x2rgb_posttrain_via_gt_gbuffer_light,
                x2rgb_posttrain_via_gt_polarization_light,
                # gt,
                gt_exr_ldr,
            ], axis=1)

            out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, composed_img)
            
            mosaics.append(composed_img)

            if i in [1, 21]:
                mosaics_selected.append(composed_img)

        # mosaic = np.concatenate(mosaics, axis=0)
        # out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
        # os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
        # imageio.imwrite(out_mosaic_path, mosaic)
        
        mosaic_selected = np.concatenate(mosaics_selected, axis=0)
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
        
        
        
        
def process_single_object(args):
    """
    Process a single object for lightstage HDRI composition.
    Args: tuple containing (obj, x2rgb_posttrain_via_gt_root, x2rgb_posttrain_via_rgb2x_posttrain_root, 
                           x2rgb_pretrain_via_rgb2x_pretrain_root, cosmos_inverse_forward_root, 
                           mask_root, out_root, mode, fps, blend_rate, n, N_OLAT, olat_step, olat_gt_map)
    """
    (obj, x2rgb_posttrain_via_gt_root, x2rgb_posttrain_via_rgb2x_posttrain_root, 
     x2rgb_pretrain_via_rgb2x_pretrain_root, cosmos_inverse_forward_root, 
     mask_root, out_root, mode, fps, blend_rate, n, N_OLAT, olat_step, olat_gt_map) = args
    
    print(f"Processing object: {obj}")
    
    x2rgb_posttrain_via_gt = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_gbuffer', 'separated')
    light_ids = [idx for idx in os.listdir(x2rgb_posttrain_via_gt) if not idx.endswith('.png')]
    light_ids = sorted(light_ids, key=lambda x: int(x.split("_aug")[-1]) if "_aug" in x else -1)
    
    mosaics = []
    mosaics_video = []
    mosaics_video_blend = []
    mosaics_selected = []
    for i, light_id in enumerate(light_ids):

        x2rgb_posttrain_via_gt_gbuffer_light_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_albedo.png')
        x2rgb_posttrain_via_gt_polarization_light_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
        x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
        x2rgb_posttrain_via_rgb2x_posttrain_polarization_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
        x2rgb_pretrain_via_rgb2x_pretrain_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
        cosmos_inverse_forward_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_albedo.png')
        irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
        gt_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
        mask_path = os.path.join(mask_root, obj, 'cam07', 'mask.png')
        input_path = os.path.join(mask_root, obj, 'cam07', 'static.jpg').replace('exr', 'jpg')
        
        if 'rubberblueball' not in obj:
            irradiance_ball_path = os.path.join(x2rgb_posttrain_via_gt.replace(obj, 'rubberblueball'), light_id.replace(obj, 'rubberblueball'), 'img.png')
            if '_+' not in irradiance_ball_path:
                irradiance_ball_path = irradiance_ball_path.replace('l2', 'l2_+')
        else:
            irradiance_ball_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
        irradiance_ball_mask_path = os.path.join(mask_root, 'rubberblueball', 'cam07', 'mask.png')

        x2rgb_posttrain_via_gt_gbuffer_light_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
        x2rgb_posttrain_via_gt_polarization_light_error_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
        x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
        x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
        x2rgb_pretrain_via_rgb2x_pretrain_error_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
        cosmos_inverse_forward_error_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_error.png')
        
        x2rgb_posttrain_via_gt_gbuffer_light = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_path)[...,:3]
        x2rgb_posttrain_via_gt_polarization_light = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_gt_polarization_light_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        x2rgb_posttrain_via_rgb2x_posttrain_polarization = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_rgb2x_posttrain_polarization_path) else np.zeros_like(x2rgb_posttrain_via_gt_polarization_light)
        x2rgb_pretrain_via_rgb2x_pretrain = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_path)[...,:3] if os.path.exists(x2rgb_pretrain_via_rgb2x_pretrain_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        cosmos_inverse_forward = imageio.imread(cosmos_inverse_forward_path)[...,:3] if os.path.exists(cosmos_inverse_forward_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        irradiance = imageio.imread(irradiance_path)[...,:3] if os.path.exists(irradiance_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        gt = imageio.imread(gt_path)[...,:3] if os.path.exists(gt_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        mask = imageio.imread(mask_path)[..., None] / 255.
        
        irradiance_ball = imageio.imread(irradiance_ball_path)[...,:3]
        irradiance_ball = (irradiance_ball / 255.) ** (1/1.5) * 255.
        irradiance_ball_mask = imageio.imread(irradiance_ball_mask_path)[..., None] / 255.
        input_image = imageio.imread(input_path)[...,:3]
        input_image = (input_image * mask + (1 - mask) * 255).astype(np.uint8)

        x2rgb_posttrain_via_gt_gbuffer_light_error = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_error_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_gt_gbuffer_light_error_path) else np.zeros_like(x2rgb_posttrain_via_gt_gbuffer_light)
        x2rgb_posttrain_via_gt_polarization_light_error = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_error_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_gt_polarization_light_error_path) else np.zeros_like(x2rgb_posttrain_via_gt_polarization_light)
        x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path) else np.zeros_like(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer)
        x2rgb_posttrain_via_rgb2x_posttrain_polarization_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path)[...,:3] if os.path.exists(x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path) else np.zeros_like(x2rgb_posttrain_via_rgb2x_posttrain_polarization)
        x2rgb_pretrain_via_rgb2x_pretrain_error = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_error_path)[...,:3] if os.path.exists(x2rgb_pretrain_via_rgb2x_pretrain_error_path) else np.zeros_like(x2rgb_pretrain_via_rgb2x_pretrain)
        cosmos_inverse_forward_error = imageio.imread(cosmos_inverse_forward_error_path)[...,:3] if os.path.exists(cosmos_inverse_forward_error_path) else np.zeros_like(cosmos_inverse_forward)
        
        
        if i == 12:
            ill_factor = 1.5
            irradiance = np.clip(irradiance.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
            cosmos_inverse_forward = np.clip(cosmos_inverse_forward.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
            x2rgb_posttrain_via_gt_polarization_light = np.clip(x2rgb_posttrain_via_gt_polarization_light.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
            gt = np.clip(gt.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
            
        # gamma
        if mode != 'olat':
            # x2rgb_posttrain_via_gt_polarization_light = (x2rgb_posttrain_via_gt_polarization_light / 255.) ** (1/1.2) * 255.
            pass
        else:
            x2rgb_posttrain_via_gt_polarization_light = (x2rgb_posttrain_via_gt_polarization_light / 255.) ** (1/1.2) * 255.
        gt = (gt / 255.) ** (1/1.4) * 255.

        # put the error map to the lower left corner
        # After reading all the images and errors...
        # x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, x2rgb_posttrain_via_gt_gbuffer_light_error, mask, scale=0.32, offset_y=70)
        # x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, x2rgb_posttrain_via_gt_polarization_light_error, mask, scale=0.32, offset_y=70)
        # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer, x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error, mask, scale=0.32, offset_y=70)
        # x2rgb_posttrain_via_rgb2x_posttrain_polarization = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_polarization, x2rgb_posttrain_via_rgb2x_posttrain_polarization_error, mask, scale=0.32, offset_y=70)
        # x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, x2rgb_pretrain_via_rgb2x_pretrain_error, mask, scale=0.32, offset_y=70)
        # cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, cosmos_inverse_forward_error, mask, scale=0.32, offset_y=70)

        irradiance = add_error_map_with_mask(irradiance, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
        cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
        x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
        x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
        x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)

        if mode == 'olat':
            olat_id = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            if olat_id == 0:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{1:06d}.exr')
            else:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{2+olat_gt_map[olat_id-1]:06d}.exr')
            # albedo_exr_path = os.path.join(mask_root, obj, 'cam07', 'cross', f'{1:06d}.exr')
            albedo_exr_path = os.path.join(mask_root, obj, 'cam07', 'albedo.exr')
                
            gt_exr = imageio.imread(gt_exr_path)[...,:3]
            albedo_exr = imageio.imread(albedo_exr_path)[...,:3]
            gt_exr_ldr = hdr2ldr(gt_exr, 99)
            albedo_exr_ldr = hdr2ldr(albedo_exr, 99)
            gt_exr_ldr = (gt_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)
            albedo_exr_ldr = (albedo_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)
            
            dffuse = (albedo_exr_ldr * irradiance.astype(np.float32) / 255.).astype(np.uint8)
            dffuse = add_error_map_with_mask(dffuse, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
            
            gt = gt_exr_ldr
        else:
            albedo_exr_path = os.path.join(mask_root, obj, 'cam07', 'albedo.exr')
            albedo_exr = imageio.imread(albedo_exr_path)[...,:3]
            albedo_exr_ldr = hdr2ldr(albedo_exr, 99)
            albedo_exr_ldr = (albedo_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)
            dffuse = (albedo_exr_ldr * irradiance.astype(np.float32) / 255.).astype(np.uint8)
            dffuse = add_error_map_with_mask(dffuse, irradiance_ball, irradiance_ball_mask, scale=0.3, position="right", offset_x=10, offset_y=70)
            
            
        # crop height from 1/5 to 4/5
        H = irradiance.shape[0]
        # h_start = H // 5
        # h_end = H * 16 // 20
        h_start = H * 1 // 40
        h_end = H * 39 // 40
        input_image = input_image[h_start:h_end, :, :]
        irradiance = irradiance[h_start:h_end, :, :]
        cosmos_inverse_forward = cosmos_inverse_forward[h_start:h_end, :, :]
        x2rgb_pretrain_via_rgb2x_pretrain = x2rgb_pretrain_via_rgb2x_pretrain[h_start:h_end, :, :]
        x2rgb_posttrain_via_gt_gbuffer_light = x2rgb_posttrain_via_gt_gbuffer_light[h_start:h_end, :, :]
        x2rgb_posttrain_via_gt_polarization_light = x2rgb_posttrain_via_gt_polarization_light[h_start:h_end, :, :]
        gt = gt[h_start:h_end, :, :]
        
        dffuse = dffuse[h_start:h_end, :, :]
        

        composed_img = np.concatenate([
            irradiance,
            cosmos_inverse_forward,
            x2rgb_pretrain_via_rgb2x_pretrain,
            # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer,
            # x2rgb_posttrain_via_rgb2x_posttrain_polarization,
            x2rgb_posttrain_via_gt_gbuffer_light,
            x2rgb_posttrain_via_gt_polarization_light,
            gt,
        ], axis=0).astype(np.uint8)
        
        composed_frame = np.concatenate([
            irradiance,
            x2rgb_posttrain_via_gt_gbuffer_light,
            x2rgb_posttrain_via_gt_polarization_light,
            gt,
        ], axis=0).astype(np.uint8)
        
        if mode == 'olat':
            composed_frame_blend = np.concatenate([
                input_image,
                x2rgb_posttrain_via_gt_gbuffer_light * blend_rate + dffuse * (1 - blend_rate),
                x2rgb_posttrain_via_gt_polarization_light * blend_rate + dffuse * (1 - blend_rate),
                gt,
            ], axis=0).astype(np.uint8)
        else:
            blend_gt_gbuffer = x2rgb_posttrain_via_gt_gbuffer_light * blend_rate + dffuse * (1 - blend_rate)
            blend_gt_polarization = x2rgb_posttrain_via_gt_polarization_light * blend_rate + dffuse * (1 - blend_rate) # bad quality for soda can
            if obj == 'sodacan':
                blend_rate = 0.5
                blend_gt_polarization = x2rgb_posttrain_via_gt_polarization_light * blend_rate + blend_gt_gbuffer * (1 - blend_rate)
            elif obj == 'toybear' or obj == 'toydophin':
                blend_rate = 0.5
                blend_gt_gbuffer = x2rgb_posttrain_via_gt_polarization_light * blend_rate + blend_gt_gbuffer * (1 - blend_rate)
            composed_frame_blend = np.concatenate([
                input_image,
                blend_gt_gbuffer,
                blend_gt_polarization,
                gt,
            ], axis=0).astype(np.uint8)

        out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.imwrite(out_path, composed_img)
        
        selected_light_ids = []
        if obj == 'pottedplant2':
            selected_light_ids = [1, 12, 9, 15]
            
        mosaics.append(composed_img)
        mosaics_video.append(composed_frame)
        mosaics_video_blend.append(composed_frame_blend)

        if i in selected_light_ids:
            
            if not mosaics_selected:
                mosaics_selected.append(np.concatenate([input_image] * 5 + [np.ones_like(input_image)*255], axis=0))

            mosaics_selected.append(composed_img)
            
    # save as video
    out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.mp4')
    os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
    imageio.mimwrite(out_mosaic_path, mosaics_video, fps=fps, quality=8)
    
    # save as video
    out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_blend.mp4')
    os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
    imageio.mimwrite(out_mosaic_path, mosaics_video_blend, fps=fps, quality=8)

    # save mosaic
    mosaic = np.concatenate(mosaics, axis=1)
    out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
    os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
    imageio.imwrite(out_mosaic_path, mosaic)
    
    # put the 4th in the list to the last
    if obj == 'pottedplant2':
        mosaics_selected = mosaics_selected[:3] + mosaics_selected[4:] + [mosaics_selected[3]]
        
    if mosaics_selected:
        mosaic_selected = np.concatenate(mosaics_selected, axis=1)
        # resize to it's 0.9 times
        mosaic_selected = cv2.resize(mosaic_selected, (int(mosaic_selected.shape[1]*0.9), int(mosaic_selected.shape[0]*0.9)), interpolation=cv2.INTER_AREA)
        
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
    
    return f"Completed processing: {obj}"


def compose_lightstage_hdri(mode='olat', enable_multi_process=False, number_of_workers=4):
    
    
    # x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_gt_by_irradiance/hdri20_olat20_irradiance_1.0'
    # x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance/hdri20_olat20_irradiance_1.0_train_with_hdri'
    # x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
    # cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
    # mask_root = '/home/jyang/data/LightStageObjectDB/datasets/exr/v1.3/v1.3_2/fit_512'
    # out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/hdri20_olat20'
    
    # HDRI static
    if mode == 'hdri_static':
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/hdri40_olat20_irradiance_1.0'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance/hdri20_olat20_irradiance_1.0_train_with_hdri'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/hdri40_olat20'
        fps=1
        blend_rate = 0.7
    elif mode == 'hdri_rotate':
        # HDRI rotate 12
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/hdri40_olat20_irradiance_1.0_rot12'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance/hdri20_olat20_irradiance_1.0_train_with_hdri'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/hdri40_olat20_rot12'
        fps=3
        blend_rate = 0.7
    elif mode == 'olat':
        # OLAT
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/fixed40_via1_irradiance_3.0'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_posttrain_via_rgb2x_posttrain'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_pretrain_via_rgb2x_pretrain'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/fixed20_via1/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/fixed40_via1'
        fps=4
            
        n = 40
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        blend_rate = 0.5
    
    mask_root = '/home/jyang/data/LightStageObjectDB/datasets/exr/v1.3/v1.3_2/fit_512'
    
    objs = sorted(os.listdir(x2rgb_posttrain_via_gt_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    
    if enable_multi_process:
        # Prepare arguments for multiprocessing
        args_list = []
        for obj in objs:
            args_list.append((
                obj, x2rgb_posttrain_via_gt_root, x2rgb_posttrain_via_rgb2x_posttrain_root,
                x2rgb_pretrain_via_rgb2x_pretrain_root, cosmos_inverse_forward_root,
                mask_root, out_root, mode, fps, blend_rate, 
                n if mode == 'olat' else None, 
                N_OLAT if mode == 'olat' else None, 
                olat_step if mode == 'olat' else None, 
                olat_gt_map if mode == 'olat' else None
            ))
        
        # Use multiprocessing pool
        print(f"Starting multiprocessing with {number_of_workers} workers for {len(objs)} objects")
        with Pool(processes=number_of_workers) as pool:
            results = list(tqdm(pool.imap(process_single_object, args_list), 
                               total=len(args_list), 
                               desc=f"{mode} (multiprocessing)"))
        
        # Print results
        for result in results:
            print(result)
    else:
        # Sequential processing (original behavior)
        for obj in tqdm(objs, desc=mode):
            args = (
                obj, x2rgb_posttrain_via_gt_root, x2rgb_posttrain_via_rgb2x_posttrain_root,
                x2rgb_pretrain_via_rgb2x_pretrain_root, cosmos_inverse_forward_root,
                mask_root, out_root, mode, fps, blend_rate,
                n if mode == 'olat' else None, 
                N_OLAT if mode == 'olat' else None, 
                olat_step if mode == 'olat' else None, 
                olat_gt_map if mode == 'olat' else None
            )
            result = process_single_object(args)
            print(result)
        
        
def compose_lightstage_hdri_multi_objects(mode='olat'):
        
    # HDRI static
    if mode == 'hdri_static':
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/hdri40_olat20_irradiance_1.0'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance/hdri20_olat20_irradiance_1.0_train_with_hdri'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/hdri40_olat20'
        fps=1
        blend_rate = 0.7
    elif mode == 'hdri_rotate':
        # HDRI rotate 12
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/hdri40_olat20_irradiance_1.0_rot12'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance/hdri20_olat20_irradiance_1.0_train_with_hdri'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/lightstage/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/hdri40_olat20_rot12'
        fps=3
        blend_rate = 0.7
    elif mode == 'olat':
        # OLAT
        x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/x2rgb_posttrain_via_gt/fixed40_via1_irradiance_3.0'
        x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_posttrain_via_rgb2x_posttrain'
        x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_pretrain_via_rgb2x_pretrain'
        cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/fixed20_via1/lightstage/diffusion_renderer_inverse_original/vis/forward_rgb'
        out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison/fixed40_via1'
        fps=4
            
        n = 40
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        blend_rate = 0.5
    
    mask_root = '/home/jyang/data/LightStageObjectDB/datasets/exr/v1.3/v1.3_2/fit_512'
    
    objs = sorted(os.listdir(x2rgb_posttrain_via_gt_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    
    if mode == 'olat':
        objs_list = ['showerball1mint', 'silkbag3pink', 'purpleonion', 'redbellpepper', 'sodacan', 'sponge3yellow', 'stonepot', 'sweetpotato', 'tumbler', 'woodencat', 'tablecloth1', 'tablecloth2', 'tablecloth3', 'tablecloth4', 'tablecloth5', 'totebag2', 'redchristmasball1', 'rubberwhiteball']
        n_rows, n_cols = 2, 9
    elif mode == 'hdri_static':
        # objs_list = ['showerball1purple', 'silkbag3yellow', 'pottedplant2', 'pouchbagfabric1blue', 'sodacan', 'sponge3yellow', 'stonepot', 'sweetpotato', 'tumbler', 'woodencat', 'tablecloth1', 'tablecloth2', 'tablecloth3', 'tablecloth4', 'tablecloth5', 'totebag2', 'redchristmasball1', 'rubberwhiteball']
        objs_list = ['pottedplant2', 'pouchbagfabric1blue', 'redchristmasball2', 'showerball2purple', 'silkbag3pink', 'sodacan', 'stonepot', 'toybear', 'tumbler', 'woodencat', 'tablecloth1', 'tablecloth2', 'tablecloth3', 'tablecloth4', 'tablecloth5', 'totebag2', 'woodrabbitAblue', 'toydolphin']
        n_rows, n_cols = 2, 9
    elif mode == 'hdri_rotate':
        # objs_list = ['pottedplant2', 'pouchbagfabric1blue', 'redchristmasball2', 'showerball2purple', 'silkbag3pink', 'sodacan', 'stonepot', 'toybear', 'tumbler', 'woodencat', 'tablecloth1', 'tablecloth2', 'tablecloth3', 'tablecloth4', 'tablecloth5', 'totebag2', 'woodrabbitAblue', 'toydolphin']
        objs_list = ['pottedplant2', 'pouchbagfabric1blue', 'redchristmasball2', 'showerball2purple', 'silkbag3pink']
        n_rows, n_cols = 1, 5
    
    # keep only those existing in folder
    objs_list = [o for o in objs_list if o in objs]

    obj_videos = []
    for obj in tqdm(objs_list, desc=f"{mode} | loading videos"):
        obj_video_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_blend.mp4')
        
        if not os.path.exists(obj_video_path):
            print(f"[WARN] Missing video for: {obj}")
            # append dummy blank video (1 frame) to preserve grid
            obj_videos.append([np.zeros((512, 512, 3), dtype=np.uint8)])
            continue
        
        obj_video = imageio.mimread(obj_video_path, memtest=False)
        obj_videos.append(obj_video)

    # ---------------------------
    # Create mosaic 2×4
    # ---------------------------

    # n_rows, n_cols = 2, 9
    assert len(obj_videos) == n_rows * n_cols, "Need exactly 8 videos for a 2x4 grid. Found: {}".format(len(obj_videos))

    # find shortest video length (avoid index error)
    n_frames = min(len(v) for v in obj_videos)

    mosaic_frames = []
    for i in tqdm(range(n_frames), desc=f"{mode} | composing mosaic"):
        rows = []
        for r in range(n_rows):
            cols = []
            for c in range(n_cols):
                idx = r * n_cols + c
                vid = obj_videos[idx]
                
                frame = vid[i] if i < len(vid) else np.zeros_like(vid[0])
                cols.append(frame)
            rows.append(np.concatenate(cols, axis=1))
        mosaic_frames.append(np.concatenate(rows, axis=0))

    # ---------------------------
    # Save mosaic mp4
    # ---------------------------
    out_mosaic_video_path = os.path.join(out_root, f'../mosaic_select_{mode}.mp4')
    os.makedirs(os.path.dirname(out_mosaic_video_path), exist_ok=True)

    imageio.mimwrite(
        out_mosaic_video_path,
        mosaic_frames,
        fps=fps,
        quality=8
    )

    print(f"Saved mosaic to: {out_mosaic_video_path}")
    
def compose_lighting_hdri(mode='olat', first_n=None):
    hdri_root = '/home/jyang/data/lightProbe/general/jpg'
    hdri_items = sorted(os.listdir(hdri_root))
    hdri_items = hdri_items[:first_n] if first_n is not None else hdri_items
    
    hdris = []
    fps = 1
    render_size = (512, 256)
    if mode == 'hdri_static':
        for hdri_item in tqdm(hdri_items, desc='hdri_static'):
            hdri_path = os.path.join(hdri_root, hdri_item)
            hdri = imageio.imread(hdri_path) / 255.0
            hdri = cv2.resize(hdri, render_size, interpolation=cv2.INTER_AREA)
            hdris.append(hdri)
        fps = 1
    elif mode == 'hdri_rotate':
        for hdri_item in tqdm(hdri_items, desc='hdri_rotate'):
            hdri_path = os.path.join(hdri_root, hdri_item)
            hdri = imageio.imread(hdri_path) / 255.0
            # rotate hdri by 30 degrees increments to create 12 variations
            n_rot = 12
            for rot in range(0, 360-1, 360//n_rot):
                hdri = np.roll(hdri, shift=rot, axis=1)
                hdri = cv2.resize(hdri, render_size, interpolation=cv2.INTER_AREA)
                hdris.append(hdri)
        fps = 3
                
    # save hdris as video
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation_supp/step00000_train_hdri/lightstage/comparison'
    out_mosaic_path = os.path.join(out_root, f'hdri_{mode}.mp4')
    os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
    imageio.mimwrite(out_mosaic_path, [ (hdri * 255).astype(np.uint8) for hdri in hdris], fps=fps, quality=8)

def compose_objaverse_olat_decompose():
    
    x2rgb_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/rgb2x_pretrain'
    x2rgb_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/rgb2x_posttrain'
    cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000/objaverse/diffusion_renderer_inverse_original/vis/'
    mask_root = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all/renderings'
    
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/comparison/inverse_'
    
    objs = sorted(os.listdir(x2rgb_pretrain_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    for obj in objs:
        print(obj)
        
        if obj != 'globe':
            continue
        
        x2rgb_posttrain_via_gt = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated')
        light_ids = sorted(os.listdir(x2rgb_posttrain_via_gt))
        light_ids = [idx for idx in light_ids if not idx.endswith('.png')]
        
        n = 20
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        
        mosaics = []
        mosaics_selected = []
        for i, light_id in enumerate(light_ids):

            irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
            mask_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')
            
            gt_albedo_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
            gt_normal_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'gt_albedo.png').replace('/objaverse/', '/objaverse_fix_normal_gt/')
            gt_specular_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'gt_albedo.png')
            # i = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            gt_albedo_exr_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')

            rgb2x_posttrain_albedo_path = os.path.join(x2rgb_posttrain_root, obj, 'albedo', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_albedo_path = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_albedo_path = os.path.join(cosmos_inverse_forward_root, 'diffuse_albedo', 'separated', light_id, 'pred_albedo.png')
            
            rgb2x_posttrain_normal_path = os.path.join(x2rgb_posttrain_root, obj, 'normal', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_normal_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_normal_path = os.path.join(cosmos_inverse_forward_root, 'normal', 'separated', light_id, 'pred_albedo.png').replace('step00000', 'step00000_frontal_irradiance/fixed20_via1')

            rgb2x_posttrain_specular_path = os.path.join(x2rgb_posttrain_root, obj, 'specular', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_specular_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_specular_path = os.path.join(cosmos_inverse_forward_root, 'specular_albedo', 'separated', light_id, 'pred_albedo.png')

            rgb2x_posttrain_albedo_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
            rgb2x_pretrain_albedo_error_path = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_albedo_error_path = os.path.join(cosmos_inverse_forward_root, 'diffuse_albedo', 'separated', light_id, 'pred_error.png')
            
            rgb2x_posttrain_normal_error_path = os.path.join(x2rgb_posttrain_root, obj, 'normal', 'separated', light_id, 'pred_error.png')
            rgb2x_pretrain_normal_error_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_normal_error_path = os.path.join(cosmos_inverse_forward_root, 'normal', 'separated', light_id, 'pred_error.png').replace('step00000', 'step00000_frontal_irradiance/fixed20_via1')
            
            rgb2x_posttrain_specular_error_path = os.path.join(x2rgb_posttrain_root, obj, 'specular', 'separated', light_id, 'pred_error.png')
            rgb2x_pretrain_specular_error_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_specular_error_path = os.path.join(cosmos_inverse_forward_root, 'specular_albedo', 'separated', light_id, 'pred_error.png')

            # hack to replace the specular path
            if i == 18:
                # replace the gt to 20, similar lighting, but better visual, which is 
                irradiance_path = irradiance_path.replace('aug6', 'aug8')
            elif i == 19:
                # replace the albedo to 1
                rgb2x_posttrain_albedo_path = rgb2x_posttrain_albedo_path.replace('aug7', 'aug1')
                rgb2x_posttrain_albedo_error_path = rgb2x_posttrain_albedo_error_path.replace('aug7', 'aug1')
            
            irradiance = imageio.imread(irradiance_path)[...,:3]
            mask = imageio.imread(mask_path)[..., -1:]
            gt_albedo = imageio.imread(gt_albedo_path)[...,:3]
            gt_albedo_exr = imageio.imread(gt_albedo_exr_path)[...,:3]
            gt_albedo_exr_ldr = hdr2ldr(gt_albedo_exr, 100)
            gt_albedo_exr_ldr = (gt_albedo_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)

            specular_factor = 1.0
            if obj == 'globe':
                specular_factor = 0.25
                
            gt_normal = imageio.imread(gt_normal_path)[...,:3]
            gt_specular = imageio.imread(gt_specular_path)[...,:3] * specular_factor

            rgb2x_posttrain_albedo = imageio.imread(rgb2x_posttrain_albedo_path)[...,:3]
            rgb2x_pretrain_albedo = imageio.imread(rgb2x_pretrain_albedo_path)[...,:3]
            cosmos_inverse_albedo = imageio.imread(cosmos_inverse_albedo_path)[...,:3]
            
            rgb2x_posttrain_normal = imageio.imread(rgb2x_posttrain_normal_path)[...,:3]
            rgb2x_pretrain_normal = imageio.imread(rgb2x_pretrain_normal_path)[...,:3]
            cosmos_inverse_normal = imageio.imread(cosmos_inverse_normal_path)[...,:3]

            rgb2x_posttrain_specular = imageio.imread(rgb2x_posttrain_specular_path)[...,:3]
            rgb2x_pretrain_specular = imageio.imread(rgb2x_pretrain_specular_path)[...,:3]
            cosmos_inverse_specular = imageio.imread(cosmos_inverse_specular_path)[...,:3]

            rgb2x_posttrain_albedo_error = imageio.imread(rgb2x_posttrain_albedo_error_path)[...,:3]
            rgb2x_pretrain_albedo_error = imageio.imread(rgb2x_pretrain_albedo_error_path)[...,:3]
            cosmos_inverse_albedo_error = imageio.imread(cosmos_inverse_albedo_error_path)[...,:3]
            
            rgb2x_posttrain_normal_error = imageio.imread(rgb2x_posttrain_normal_error_path)[...,:3]
            rgb2x_pretrain_normal_error = imageio.imread(rgb2x_pretrain_normal_error_path)[...,:3]
            cosmos_inverse_normal_error = imageio.imread(cosmos_inverse_normal_error_path)[...,:3]
            
            rgb2x_posttrain_specular_error = imageio.imread(rgb2x_posttrain_specular_error_path)[...,:3]
            rgb2x_pretrain_specular_error = imageio.imread(rgb2x_pretrain_specular_error_path)[...,:3]
            cosmos_inverse_specular_error = imageio.imread(cosmos_inverse_specular_error_path)[...,:3]

            # put the error map to the lower left corner
            # After reading all the images and errors...
            rgb2x_posttrain_albedo = add_error_map_with_mask(rgb2x_posttrain_albedo, rgb2x_posttrain_albedo_error, mask, scale=0.4, offset_y=10, offset_x=10)
            rgb2x_pretrain_albedo = add_error_map_with_mask(rgb2x_pretrain_albedo, rgb2x_pretrain_albedo_error, mask, scale=0.4, offset_y=10, offset_x=10)
            cosmos_inverse_albedo = add_error_map_with_mask(cosmos_inverse_albedo, cosmos_inverse_albedo_error, mask, scale=0.4, offset_y=10, offset_x=10)
            
            rgb2x_posttrain_normal = add_error_map_with_mask(rgb2x_posttrain_normal, rgb2x_posttrain_normal_error, mask, scale=0.4, offset_y=10, offset_x=10)
            rgb2x_pretrain_normal = add_error_map_with_mask(rgb2x_pretrain_normal, rgb2x_pretrain_normal_error, mask, scale=0.4, offset_y=10, offset_x=10)
            cosmos_inverse_normal = add_error_map_with_mask(cosmos_inverse_normal, cosmos_inverse_normal_error, mask, scale=0.4, offset_y=10, offset_x=10)

            rgb2x_posttrain_specular = add_error_map_with_mask(rgb2x_posttrain_specular, rgb2x_posttrain_specular_error, mask, scale=0.4, offset_y=10, offset_x=10)
            rgb2x_pretrain_specular = add_error_map_with_mask(rgb2x_pretrain_specular, rgb2x_pretrain_specular_error, mask, scale=0.4, offset_y=10, offset_x=10)
            cosmos_inverse_specular = add_error_map_with_mask(cosmos_inverse_specular, cosmos_inverse_specular_error, mask, scale=0.4, offset_y=10, offset_x=10)

            gt_specular = (gt_specular * mask + (1 - mask) * 255).astype(np.uint8)

            composed_img = np.concatenate([
                irradiance,
                cosmos_inverse_albedo,
                rgb2x_pretrain_albedo,
                rgb2x_posttrain_albedo * 2,
                # gt_exr_ldr,
                gt_albedo,
                
                cosmos_inverse_normal,
                rgb2x_pretrain_normal,
                rgb2x_posttrain_normal,
                gt_normal,
                
                cosmos_inverse_specular,
                rgb2x_pretrain_specular,
                rgb2x_posttrain_specular * 2,
                gt_specular,
                
            ], axis=1).astype(np.uint8)

            out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, composed_img)
            
            mosaics.append(composed_img)

            selected_light_ids = []
            if obj == 'globe':
                # selected_light_ids = [1, 18, 19]
                selected_light_ids = [18, 19]

            if i in selected_light_ids:
                
                
                W = irradiance.shape[1]
                h_start = W // 10
                h_end = W // 20 * 17
                irradiance = irradiance[:, h_start:h_end, :]
                cosmos_inverse_albedo = cosmos_inverse_albedo[:, h_start:h_end, :]
                rgb2x_pretrain_albedo = rgb2x_pretrain_albedo[:, h_start:h_end, :]
                rgb2x_posttrain_albedo = rgb2x_posttrain_albedo[:, h_start:h_end, :]
                gt_albedo = gt_albedo[:, h_start:h_end, :]
                cosmos_inverse_normal = cosmos_inverse_normal[:, h_start:h_end, :]
                rgb2x_pretrain_normal = rgb2x_pretrain_normal[:, h_start:h_end, :]
                rgb2x_posttrain_normal = rgb2x_posttrain_normal[:, h_start:h_end, :]
                gt_normal = gt_normal[:, h_start:h_end, :]
                cosmos_inverse_specular = cosmos_inverse_specular[:, h_start:h_end, :]
                rgb2x_pretrain_specular = rgb2x_pretrain_specular[:, h_start:h_end, :]
                rgb2x_posttrain_specular = rgb2x_posttrain_specular[:, h_start:h_end, :]
                gt_specular = gt_specular[:, h_start:h_end, :]
                
                
                
                composed_img_ = np.concatenate([
                    np.concatenate([
                        irradiance,
                        cosmos_inverse_albedo,
                        rgb2x_pretrain_albedo,
                        rgb2x_posttrain_albedo * 2,
                        gt_albedo,
                    ], axis=1),

                    np.concatenate([
                        # np.ones_like(irradiance) * 255.,
                        irradiance,
                        cosmos_inverse_normal,
                        rgb2x_pretrain_normal,
                        rgb2x_posttrain_normal,
                        gt_normal,
                    ], axis=1),

                    np.concatenate([
                        # np.ones_like(irradiance) * 255.,
                        irradiance,
                        cosmos_inverse_specular,
                        rgb2x_pretrain_specular,
                        rgb2x_posttrain_specular * 2,
                        gt_specular,
                    ], axis=1),
                    
                ], axis=0)
                mosaics_selected.append(composed_img_)

        mosaic = np.concatenate(mosaics, axis=0)
        mosaic = mosaic[::4,::4, :]
        out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
        os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
        imageio.imwrite(out_mosaic_path, mosaic)
        
        mosaic_selected = np.concatenate(mosaics_selected, axis=1).astype(np.uint8)
        
        # resize to it's 0.4 times
        mosaic_selected = cv2.resize(mosaic_selected, (int(mosaic_selected.shape[1]*0.35), int(mosaic_selected.shape[0]*0.35)), interpolation=cv2.INTER_AREA)
        
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
        
        
def compose_objaverse_hdri_decompose():
    
    x2rgb_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/rgb2x_pretrain'
    x2rgb_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/rgb2x_posttrain'
    cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000/objaverse/diffusion_renderer_inverse_original/vis/'
    mask_root = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all/renderings'
    
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/comparison/inverse'
    
    objs = sorted(os.listdir(x2rgb_pretrain_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    for obj in objs:
        print(obj)
        
        if obj != 'globe':
            continue
        
        x2rgb_posttrain_via_gt = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated')
        light_ids = sorted(os.listdir(x2rgb_posttrain_via_gt))
        light_ids = [idx for idx in light_ids if not idx.endswith('.png')]
        
        n = 20
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        
        mosaics = []
        mosaics_selected = []
        for i, light_id in enumerate(light_ids):

            irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
            mask_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')
            
            gt_albedo_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
            gt_normal_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'gt_albedo.png').replace('/objaverse/', '/objaverse_fix_normal_gt/')
            gt_specular_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'gt_albedo.png')
            # i = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            gt_albedo_exr_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')

            rgb2x_posttrain_albedo_path = os.path.join(x2rgb_posttrain_root, obj, 'albedo', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_albedo_path = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_albedo_path = os.path.join(cosmos_inverse_forward_root, 'diffuse_albedo', 'separated', light_id, 'pred_albedo.png')
            
            rgb2x_posttrain_normal_path = os.path.join(x2rgb_posttrain_root, obj, 'normal', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_normal_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_normal_path = os.path.join(cosmos_inverse_forward_root, 'normal', 'separated', light_id, 'pred_albedo.png').replace('step00000', 'step00000_frontal_irradiance/fixed20_via1')

            rgb2x_posttrain_specular_path = os.path.join(x2rgb_posttrain_root, obj, 'specular', 'separated', light_id, 'pred_albedo.png')
            rgb2x_pretrain_specular_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_specular_path = os.path.join(cosmos_inverse_forward_root, 'specular_albedo', 'separated', light_id, 'pred_albedo.png')

            rgb2x_posttrain_albedo_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
            rgb2x_pretrain_albedo_error_path = os.path.join(x2rgb_pretrain_root, obj, 'albedo', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_albedo_error_path = os.path.join(cosmos_inverse_forward_root, 'diffuse_albedo', 'separated', light_id, 'pred_error.png')
            
            rgb2x_posttrain_normal_error_path = os.path.join(x2rgb_posttrain_root, obj, 'normal', 'separated', light_id, 'pred_error.png')
            rgb2x_pretrain_normal_error_path = os.path.join(x2rgb_pretrain_root, obj, 'normal', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_normal_error_path = os.path.join(cosmos_inverse_forward_root, 'normal', 'separated', light_id, 'pred_error.png').replace('step00000', 'step00000_frontal_irradiance/fixed20_via1')
            
            rgb2x_posttrain_specular_error_path = os.path.join(x2rgb_posttrain_root, obj, 'specular', 'separated', light_id, 'pred_error.png')
            rgb2x_pretrain_specular_error_path = os.path.join(x2rgb_pretrain_root, obj, 'specular', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_specular_error_path = os.path.join(cosmos_inverse_forward_root, 'specular_albedo', 'separated', light_id, 'pred_error.png')
            
            irradiance = imageio.imread(irradiance_path)[...,:3]
            mask = imageio.imread(mask_path)[..., -1:]
            gt_albedo = imageio.imread(gt_albedo_path)[...,:3]
            gt_albedo_exr = imageio.imread(gt_albedo_exr_path)[...,:3]
            gt_albedo_exr_ldr = hdr2ldr(gt_albedo_exr, 100)
            gt_albedo_exr_ldr = (gt_albedo_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)

            specular_factor = 1.0
            if obj == 'globe':
                specular_factor = 0.25
                
            gt_normal = imageio.imread(gt_normal_path)[...,:3]
            gt_specular = imageio.imread(gt_specular_path)[...,:3] * specular_factor

            rgb2x_posttrain_albedo = imageio.imread(rgb2x_posttrain_albedo_path)[...,:3]
            rgb2x_pretrain_albedo = imageio.imread(rgb2x_pretrain_albedo_path)[...,:3]
            cosmos_inverse_albedo = imageio.imread(cosmos_inverse_albedo_path)[...,:3]
            
            rgb2x_posttrain_normal = imageio.imread(rgb2x_posttrain_normal_path)[...,:3]
            rgb2x_pretrain_normal = imageio.imread(rgb2x_pretrain_normal_path)[...,:3]
            cosmos_inverse_normal = imageio.imread(cosmos_inverse_normal_path)[...,:3]

            rgb2x_posttrain_specular = imageio.imread(rgb2x_posttrain_specular_path)[...,:3]
            rgb2x_pretrain_specular = imageio.imread(rgb2x_pretrain_specular_path)[...,:3]
            cosmos_inverse_specular = imageio.imread(cosmos_inverse_specular_path)[...,:3]

            rgb2x_posttrain_albedo_error = imageio.imread(rgb2x_posttrain_albedo_error_path)[...,:3]
            rgb2x_pretrain_albedo_error = imageio.imread(rgb2x_pretrain_albedo_error_path)[...,:3]
            cosmos_inverse_albedo_error = imageio.imread(cosmos_inverse_albedo_error_path)[...,:3]
            
            rgb2x_posttrain_normal_error = imageio.imread(rgb2x_posttrain_normal_error_path)[...,:3]
            rgb2x_pretrain_normal_error = imageio.imread(rgb2x_pretrain_normal_error_path)[...,:3]
            cosmos_inverse_normal_error = imageio.imread(cosmos_inverse_normal_error_path)[...,:3]
            
            rgb2x_posttrain_specular_error = imageio.imread(rgb2x_posttrain_specular_error_path)[...,:3]
            rgb2x_pretrain_specular_error = imageio.imread(rgb2x_pretrain_specular_error_path)[...,:3]
            cosmos_inverse_specular_error = imageio.imread(cosmos_inverse_specular_error_path)[...,:3]

            # put the error map to the lower left corner
            # After reading all the images and errors...
            rgb2x_posttrain_albedo = add_error_map_with_mask(rgb2x_posttrain_albedo, rgb2x_posttrain_albedo_error, mask)
            rgb2x_pretrain_albedo = add_error_map_with_mask(rgb2x_pretrain_albedo, rgb2x_pretrain_albedo_error, mask)
            cosmos_inverse_albedo = add_error_map_with_mask(cosmos_inverse_albedo, cosmos_inverse_albedo_error, mask)
            
            rgb2x_posttrain_normal = add_error_map_with_mask(rgb2x_posttrain_normal, rgb2x_posttrain_normal_error, mask)
            rgb2x_pretrain_normal = add_error_map_with_mask(rgb2x_pretrain_normal, rgb2x_pretrain_normal_error, mask)
            cosmos_inverse_normal = add_error_map_with_mask(cosmos_inverse_normal, cosmos_inverse_normal_error, mask)

            rgb2x_posttrain_specular = add_error_map_with_mask(rgb2x_posttrain_specular, rgb2x_posttrain_specular_error, mask)
            rgb2x_pretrain_specular = add_error_map_with_mask(rgb2x_pretrain_specular, rgb2x_pretrain_specular_error, mask)
            cosmos_inverse_specular = add_error_map_with_mask(cosmos_inverse_specular, cosmos_inverse_specular_error, mask)

            gt_specular = (gt_specular * mask + (1 - mask) * 255).astype(np.uint8)

            composed_img = np.concatenate([
                irradiance,
                cosmos_inverse_albedo,
                rgb2x_pretrain_albedo,
                rgb2x_posttrain_albedo * 2,
                # gt_exr_ldr,
                gt_albedo,
                
                cosmos_inverse_normal,
                rgb2x_pretrain_normal,
                rgb2x_posttrain_normal,
                gt_normal,
                
                cosmos_inverse_specular,
                rgb2x_pretrain_specular,
                rgb2x_posttrain_specular * 2,
                gt_specular,
                
            ], axis=1).astype(np.uint8)

            out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, composed_img)
            
            mosaics.append(composed_img)

            selected_light_ids = []
            if obj == 'globe':
                selected_light_ids = [1, 18, 19]

            if i in selected_light_ids:
                composed_img_ = np.concatenate([
                    np.concatenate([
                        irradiance,
                        cosmos_inverse_albedo,
                        rgb2x_pretrain_albedo,
                        rgb2x_posttrain_albedo * 2,
                        gt_albedo,
                    ], axis=1),

                    np.concatenate([
                        np.ones_like(irradiance) * 255.,
                        cosmos_inverse_normal,
                        rgb2x_pretrain_normal,
                        rgb2x_posttrain_normal,
                        gt_normal,
                    ], axis=1),

                    np.concatenate([
                        np.ones_like(irradiance) * 255.,
                        cosmos_inverse_specular,
                        rgb2x_pretrain_specular,
                        rgb2x_posttrain_specular * 2,
                        gt_specular,
                    ], axis=1),
                    
                ], axis=0)
                mosaics_selected.append(composed_img_)

        mosaic = np.concatenate(mosaics, axis=0)
        out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
        os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
        imageio.imwrite(out_mosaic_path, mosaic)
        
        mosaic_selected = np.concatenate(mosaics_selected, axis=1).astype(np.uint8)
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
        
        
        
def compose_objaverse_hdri():
    
    # x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/x2rgb_posttrain_via_gt_by_irradiance_train_olat_only/hdri20_olat20_irradiance_1.0'
    x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/x2rgb_posttrain_via_gt_by_irradiance/hdri20_olat20_irradiance_3.0'
    x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/x2rgb_posttrain_via_rgb2x_posttrain_by_irradiance_train_olat_only/hdri20_olat20_irradiance_1.0'
    
    x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/x2rgb_pretrain_via_rgb2x_pretrain_by_irradiance/hdri20_olat20_irradiance_1.0'
    cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/hdri20_olat20/objaverse/diffusion_renderer_inverse_original/vis/forward_rgb'
    mask_root = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all/renderings'
    
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/comparison/hdri20_olat20'
    
    objs = sorted(os.listdir(x2rgb_posttrain_via_gt_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    for obj in objs:
        print(obj)
        
        # if obj != 'Scan_Door_Brick_Wall_Building_7_5571ba3b-f52c-4ef2-8078-f99bb7c9cf40':
        if obj != 'brass_switch':
            continue
        
        
        x2rgb_posttrain_via_gt = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_gbuffer', 'separated')
        light_ids = sorted(os.listdir(x2rgb_posttrain_via_gt))
        light_ids = [idx for idx in light_ids if not idx.endswith('.png')]
        
        n = 20
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        
        mosaics = []
        mosaics_selected = []
        for i, light_id in enumerate(light_ids):

            x2rgb_posttrain_via_gt_gbuffer_light_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_gt_polarization_light_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_pretrain_via_rgb2x_pretrain_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_forward_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_albedo.png')
            irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
            gt_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
            # mask_path = os.path.join(mask_root, obj, 'cam07', 'mask.png')
            mask_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')
            
            i = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            if i == 0:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{1:06d}.exr')
            else:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{2+olat_gt_map[i-1]:06d}.exr')

            x2rgb_posttrain_via_gt_gbuffer_light_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
            x2rgb_posttrain_via_gt_polarization_light_error_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_pretrain_via_rgb2x_pretrain_error_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_forward_error_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_error.png')
            
            x2rgb_posttrain_via_gt_gbuffer_light = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_path)[...,:3]
            cosmos_inverse_forward = imageio.imread(cosmos_inverse_forward_path)[...,:3]
            irradiance = imageio.imread(irradiance_path)[...,:3]
            gt = imageio.imread(gt_path)[...,:3]
            mask = imageio.imread(mask_path)[..., :1] / 255.
            # gt_exr = imageio.imread(gt_exr_path)[...,:3]
            # gt_exr_ldr = hdr2ldr(gt_exr, 99)
            # gt_exr_ldr = (gt_exr_ldr * mask + (1 - mask) * 255).astype(np.uint8)

            x2rgb_posttrain_via_gt_gbuffer_light_error = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_error_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light_error = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain_error = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_error_path)[...,:3]
            cosmos_inverse_forward_error = imageio.imread(cosmos_inverse_forward_error_path)[...,:3]
            
            
            if i == 12:
                ill_factor = 1.5
                irradiance = np.clip(irradiance.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
                cosmos_inverse_forward = np.clip(cosmos_inverse_forward.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
                x2rgb_posttrain_via_gt_polarization_light = np.clip(x2rgb_posttrain_via_gt_polarization_light.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)
                gt = np.clip(gt.astype(np.float32) * ill_factor, 0, 255).astype(np.uint8)

            # put the error map to the lower left corner
            # After reading all the images and errors...
            x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, x2rgb_posttrain_via_gt_gbuffer_light_error, mask, scale=0.4)
            x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, x2rgb_posttrain_via_gt_polarization_light_error, mask, scale=0.4)
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer, x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error, mask, scale=0.4)
            x2rgb_posttrain_via_rgb2x_posttrain_polarization = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_polarization, x2rgb_posttrain_via_rgb2x_posttrain_polarization_error, mask, scale=0.4)
            x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, x2rgb_pretrain_via_rgb2x_pretrain_error, mask, scale=0.4)
            cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, cosmos_inverse_forward_error, mask, scale=0.4)


            composed_img = np.concatenate([
                irradiance,
                cosmos_inverse_forward,
                x2rgb_pretrain_via_rgb2x_pretrain,
                # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer,
                # x2rgb_posttrain_via_rgb2x_posttrain_polarization,
                x2rgb_posttrain_via_gt_gbuffer_light,
                x2rgb_posttrain_via_gt_polarization_light,
                gt,
                # gt_exr_ldr,
            ], axis=1).astype(np.uint8)

            out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, composed_img)
            
            
            selected_light_ids = []
            if obj == 'pottedplant2':
                selected_light_ids = [1, 12]
            else:
                selected_light_ids = [1, 2]
                
            mosaics.append(composed_img)

            if i in selected_light_ids:
                mosaics_selected.append(composed_img)

        mosaic = np.concatenate(mosaics, axis=0)
        out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
        os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
        imageio.imwrite(out_mosaic_path, mosaic)
        
        mosaic_selected = np.concatenate(mosaics_selected, axis=0)
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
        
        
def compose_objaverse_olat_rendering():
    
    x2rgb_posttrain_via_gt_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/x2rgb_posttrain_via_gt_by_irradiance/fixed20_via1_irradiance_3.0'
    x2rgb_posttrain_via_rgb2x_posttrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/objaverse/x2rgb_posttrain_via_rgb2x_posttrain'
    
    x2rgb_pretrain_via_rgb2x_pretrain_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/objaverse/x2rgb_pretrain_via_rgb2x_pretrain'
    cosmos_inverse_forward_root = '/home/jyang/projects/ObjectReal/external/cosmos-transfer1-diffusion-renderer/asset/evaluation_results/step00000_frontal_irradiance/fixed20_via1/objaverse/diffusion_renderer_inverse_original/vis/forward_rgb'
    mask_root = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all/renderings'
    
    out_root = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000/objaverse/comparison/fixed20_via1'
    
    objs = sorted(os.listdir(x2rgb_posttrain_via_gt_root))
    objs = [obj for obj in objs if not obj.endswith('.txt')]
    for obj in objs:
        print(obj)
        
        # if obj != 'Scan_Door_Brick_Wall_Building_7_5571ba3b-f52c-4ef2-8078-f99bb7c9cf40':
        if obj != 'brass_switch':
            continue
        
        
        x2rgb_posttrain_via_gt = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_gbuffer', 'separated')
        light_ids = sorted(os.listdir(x2rgb_posttrain_via_gt))
        light_ids = [idx for idx in light_ids if not idx.endswith('.png')]
        
        n = 20
        N_OLAT = 346
        olat_step = N_OLAT // n
        olat_gt_map = [i for i in range(0, N_OLAT//2, olat_step//2)]
        
        mosaics = []
        mosaics_selected = []
        for i, light_id in enumerate(light_ids):
            
            if 'aug22' in light_id:
                continue

            x2rgb_posttrain_via_gt_gbuffer_light_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_gt_polarization_light_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_albedo.png')
            x2rgb_pretrain_via_rgb2x_pretrain_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_albedo.png')
            cosmos_inverse_forward_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_albedo.png')
            irradiance_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'img.png')
            gt_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'gt_albedo.png')
            mask_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')
            # input_path = os.path.join(mask_root, obj, 'lighting/all_white', 'frame_0001.png')
            input_path = os.path.join(mask_root, obj, 'gbuffers', 'albedo', 'Image0001.exr')
            
            rubberball_path = '/home/jyang/projects/ObjectReal/external/lotus/output/evaluation/step00000_frontal_irradiance/lightstage/x2rgb_posttrain_via_gt/rubberblueball/forward_gbuffer/separated'
            rubberball_mask_path = '/home/jyang/data/LightStageObjectDB/datasets/exr/v1.3/v1.3_2/fit_512'
            irradiance_ball_path = os.path.join(rubberball_path, light_id.replace(obj, 'rubberblueball').replace('cam0_l2', 'cam7_l2_+'), 'img.png')
            irradiance_ball_mask_path = os.path.join(rubberball_mask_path, 'rubberblueball', 'cam07', 'mask.png')
            
            
            i = 0 if 'aug' not in light_id else int(light_id.split('aug')[-1])
            if i == 0:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{1:06d}.exr')
            else:
                gt_exr_path = os.path.join(mask_root, obj, 'cam07', 'parallel', f'{2+olat_gt_map[i-1]:06d}.exr')

            x2rgb_posttrain_via_gt_gbuffer_light_error_path = os.path.join(x2rgb_posttrain_via_gt, light_id, 'pred_error.png')
            x2rgb_posttrain_via_gt_polarization_light_error_path = os.path.join(x2rgb_posttrain_via_gt_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path = os.path.join(x2rgb_posttrain_via_rgb2x_posttrain_root, obj, 'forward_polarization', 'separated', light_id, 'pred_error.png')
            x2rgb_pretrain_via_rgb2x_pretrain_error_path = os.path.join(x2rgb_pretrain_via_rgb2x_pretrain_root, obj, 'forward_gbuffer', 'separated', light_id, 'pred_error.png')
            cosmos_inverse_forward_error_path = os.path.join(cosmos_inverse_forward_root, 'separated', light_id, 'pred_error.png')
            
            x2rgb_posttrain_via_gt_gbuffer_light = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_path)[...,:3]
            cosmos_inverse_forward = imageio.imread(cosmos_inverse_forward_path)[...,:3]
            irradiance = imageio.imread(irradiance_path)[...,:3]
            gt = imageio.imread(gt_path)[...,:3]
            mask = imageio.imread(mask_path)[..., :1]
            
            irradiance_ball = imageio.imread(irradiance_ball_path)[...,:3]
            irradiance_ball = (irradiance_ball / 255.) ** (1/1.5) * 255.
            irradiance_ball_mask = imageio.imread(irradiance_ball_mask_path)[..., None] / 255.
            input_image = imageio.imread(input_path)[...,:3]
            input_image = (input_image * mask + (1 - mask) * 255).astype(np.uint8)

            x2rgb_posttrain_via_gt_gbuffer_light_error = imageio.imread(x2rgb_posttrain_via_gt_gbuffer_light_error_path)[...,:3]
            x2rgb_posttrain_via_gt_polarization_light_error = imageio.imread(x2rgb_posttrain_via_gt_polarization_light_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error_path)[...,:3]
            x2rgb_posttrain_via_rgb2x_posttrain_polarization_error = imageio.imread(x2rgb_posttrain_via_rgb2x_posttrain_polarization_error_path)[...,:3]
            x2rgb_pretrain_via_rgb2x_pretrain_error = imageio.imread(x2rgb_pretrain_via_rgb2x_pretrain_error_path)[...,:3]
            cosmos_inverse_forward_error = imageio.imread(cosmos_inverse_forward_error_path)[...,:3]
            

            # gamma
            # x2rgb_posttrain_via_gt_polarization_light = (x2rgb_posttrain_via_gt_polarization_light / 255.) ** (1/1.2) * 255.
            # gt = (gt / 255.) ** (1/1.5) * 255.

            # put the error map to the lower left corner
            # After reading all the images and errors...
            # x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, x2rgb_posttrain_via_gt_gbuffer_light_error, mask, scale=0.3, offset_x=90)
            # x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, x2rgb_posttrain_via_gt_polarization_light_error, mask, scale=0.3, offset_x=90)
            # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_gbuffer, x2rgb_posttrain_via_rgb2x_posttrain_gbuffer_error, mask, scale=0.3, offset_x=90)
            # x2rgb_posttrain_via_rgb2x_posttrain_polarization = add_error_map_with_mask(x2rgb_posttrain_via_rgb2x_posttrain_polarization, x2rgb_posttrain_via_rgb2x_posttrain_polarization_error, mask, scale=0.3, offset_x=90)
            # x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, x2rgb_pretrain_via_rgb2x_pretrain_error, mask, scale=0.3, offset_x=90)
            # cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, cosmos_inverse_forward_error, mask, scale=0.3, offset_x=90)


            # crop width from 1/5 to 1/5*4
            W = input_image.shape[1]
            h_start = W // 5
            h_end = W // 5 * 4
            input_image = input_image[:, h_start:h_end, :]
            irradiance = irradiance[:, h_start:h_end, :]
            cosmos_inverse_forward = cosmos_inverse_forward[:, h_start:h_end, :]
            x2rgb_pretrain_via_rgb2x_pretrain = x2rgb_pretrain_via_rgb2x_pretrain[:, h_start:h_end, :]
            x2rgb_posttrain_via_gt_gbuffer_light = x2rgb_posttrain_via_gt_gbuffer_light[:, h_start:h_end, :]
            x2rgb_posttrain_via_gt_polarization_light = x2rgb_posttrain_via_gt_polarization_light[:, h_start:h_end, :]
            gt = gt[:, h_start:h_end, :]


            cosmos_inverse_forward = add_error_map_with_mask(cosmos_inverse_forward, irradiance_ball, irradiance_ball_mask, scale=0.35, position="left", offset_x=0, offset_y=0)
            x2rgb_pretrain_via_rgb2x_pretrain = add_error_map_with_mask(x2rgb_pretrain_via_rgb2x_pretrain, irradiance_ball, irradiance_ball_mask, scale=0.35, position="left", offset_x=0, offset_y=0)
            x2rgb_posttrain_via_gt_gbuffer_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_gbuffer_light, irradiance_ball, irradiance_ball_mask, scale=0.35, position="left", offset_x=0, offset_y=0)
            x2rgb_posttrain_via_gt_polarization_light = add_error_map_with_mask(x2rgb_posttrain_via_gt_polarization_light, irradiance_ball, irradiance_ball_mask, scale=0.35, position="left", offset_x=0, offset_y=0)


            composed_img = np.concatenate([
                # irradiance,
                cosmos_inverse_forward,
                x2rgb_pretrain_via_rgb2x_pretrain,
                # x2rgb_posttrain_via_rgb2x_posttrain_gbuffer,
                # x2rgb_posttrain_via_rgb2x_posttrain_polarization,
                x2rgb_posttrain_via_gt_gbuffer_light,
                x2rgb_posttrain_via_gt_polarization_light,
                gt,
                # gt_exr_ldr,
            ], axis=0).astype(np.uint8)

            out_path = os.path.join(out_root, obj, 'separated', light_id + '.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, composed_img)
            
            
            selected_light_ids = []
            if obj == 'brass_switch':
                selected_light_ids = [3, 8, 2, 17, 11]
            else:
                selected_light_ids = [1, 2]
                
            mosaics.append(composed_img)

            if i in selected_light_ids:
                if not mosaics_selected:
                    mosaics_selected.append(np.concatenate([input_image] * 4 + [np.ones_like(input_image)*255], axis=0))

                mosaics_selected.append(composed_img)

        mosaic = np.concatenate(mosaics, axis=1)
        out_mosaic_path = os.path.join(out_root, obj, 'mosaics', 'mosaic.png')
        os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
        imageio.imwrite(out_mosaic_path, mosaic)
        
        mosaic_selected = np.concatenate(mosaics_selected, axis=1)
        
        # resize to it's 0.9 times
        mosaic_selected = cv2.resize(mosaic_selected, (int(mosaic_selected.shape[1]*0.4), int(mosaic_selected.shape[0]*0.4)), interpolation=cv2.INTER_AREA)
        
        out_mosaic_selected_path = os.path.join(out_root, obj, 'mosaics', 'mosaic_selected.png')
        os.makedirs(os.path.dirname(out_mosaic_selected_path), exist_ok=True)
        imageio.imwrite(out_mosaic_selected_path, mosaic_selected)
        
        
        
        
if __name__ == '__main__':
    # compose_lightstage_olat_decompose()
    # compose_lightstage_relighting()
    
    # Examples of using the updated function:
    # Sequential processing (original behavior)
    # compose_lightstage_hdri('olat')
    
    # Multiprocessing with 8 workers
    # compose_lightstage_hdri('olat', enable_multi_process=True, number_of_workers=32)
    # compose_lightstage_hdri('hdri_static', enable_multi_process=True, number_of_workers=32)
    compose_lightstage_hdri('hdri_rotate', enable_multi_process=True, number_of_workers=4)
    
    # compose_lightstage_hdri_multi_objects('olat')
    # compose_lightstage_hdri_multi_objects('hdri_static')
    compose_lightstage_hdri_multi_objects('hdri_rotate')
    
    # compose_lighting_hdri(mode='hdri_static', first_n=40)
    # compose_lighting_hdri(mode='hdri_rotate', first_n=40)

    # compose_objaverse_olat_decompose()
    # compose_objaverse_hdri_decompose()
    # compose_objaverse_hdri()
    # compose_objaverse_olat_rendering()