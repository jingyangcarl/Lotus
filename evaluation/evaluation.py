
import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, resize, InterpolationMode
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

from .dataset_depth import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from .dataset_normal.normal_dataloader import *
from .util import metric
from .util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from .util.metric import MetricTracker
from .util import normal_utils

from utils.image_utils import concatenate_images, colorize_depth_map, resize_max_res
import utils.visualize as vis_utils
from accelerate import Accelerator, PartialState

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
    # "pixel_mean",
    # "pixel_var"
]

# Referred to Marigold
def evaluation_depth(output_dir, dataset_config, base_data_dir, eval_mode, pred_suffix="", 
                     alignment="least_square", alignment_max_res=None, prediction_dir=None, 
                     gen_prediction=None, pipeline=None, save_pred=False, save_pred_vis=False,
                     processing_res=None,
                     ):
    '''
    if eval_mode == "load_prediction": assert prediction_dir is not None
    elif eval_mode == "generate_prediction": assert gen_prediction is not None and pipeline is not None
    '''
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    processing_res = processing_res or cfg_data.get('processing_res',None)
    logger.info(f"processing_res: {processing_res}")

    alignment_max_res = cfg_data.get('alignment_max_res', None)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metric file head --------------------
    per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    if save_pred_vis:
        save_vis_path = os.path.join(output_dir, "vis")
        os.makedirs(save_vis_path, exist_ok=True)

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc=f"Evaluating {cfg_data.name}"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        depth_raw_ts = depth_raw_ts.to(device)
        valid_mask_ts = valid_mask_ts.to(device)

        # Get predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(
            rgb_basename, dataset.name_mode, suffix=pred_suffix
        )
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
        if eval_mode == "load_prediction":
            pred_path = os.path.join(prediction_dir, pred_name)
            depth_pred = np.load(pred_path)
            if not os.path.exists(pred_path):
                logging.warn(f"Can't find prediction: {pred_path}")
                continue

        elif eval_mode == "generate_prediction":
            # resize to processing_res
            input_size = data["rgb_int"].shape
            if processing_res is not None:
                input_rgb =  resize_max_res(
                    data["rgb_int"],
                    max_edge_resolution=processing_res,
                    # resample_method=resample_method,
                )
            else:
                input_rgb = data["rgb_int"]
            depth_pred = gen_prediction(input_rgb, pipeline)
            # resize to original res
            if processing_res is not None:
                depth_pred = torch.tensor(depth_pred).unsqueeze(0).unsqueeze(0)
                # depth_pred = resize(depth_pred, input_size[-2:], InterpolationMode.NEAREST, antialias=True, )
                depth_pred = resize(depth_pred, input_size[-2:], antialias=True, )
                depth_pred = depth_pred.squeeze().numpy()
            

            if save_pred:
                # save_npy
                npy_dir = os.path.join(prediction_dir, 'pred_npy', cfg_data.name)
                scene_dir = os.path.join(npy_dir, os.path.dirname(rgb_name))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)
                pred_basename = get_pred_name(
                    rgb_basename, dataset.name_mode, suffix=".npy"
                )
                save_to = os.path.join(scene_dir, pred_basename)
                if os.path.exists(save_to):
                    logging.warning(f"Existing file: '{save_to}' will be overwritten")
                np.save(save_to, depth_pred)

                # save_color
                color_dir = os.path.join(prediction_dir, 'pred_color',  cfg_data.name)
                scene_dir = os.path.join(color_dir, os.path.dirname(rgb_name))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)
                pred_basename = get_pred_name(
                    rgb_basename, dataset.name_mode, suffix=".png"
                )
                save_to = os.path.join(scene_dir, pred_basename)
                if os.path.exists(save_to):
                    logging.warning(f"Existing file: '{save_to}' will be overwritten")
                depth_colored = colorize_depth_map(depth_pred)
                depth_colored.save(save_to)



        # Align with GT using least square
        if "least_square" == alignment:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
        elif "least_square_disparity" == alignment:
            # convert GT depth -> GT disparity
            gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_raw, return_mask=True
            )
            # LS alignment in disparity space
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
            # convert to depth
            disparity_pred = np.clip(
                disparity_pred, a_min=1e-3, a_max=None
            )  # avoid 0 disparity
            depth_pred = disparity2depth(disparity_pred)

        # Clip to dataset min max
        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)
        if save_pred_vis:
            depth_pred_vis = colorize_depth_map(depth_pred_ts.cpu())
            save_path = os.path.join(save_vis_path, f"{pred_name.replace('/', '_')}.png")
            depth_pred_vis.save(save_path)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(pred_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    if alignment:
        metrics_filename += f"-{alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
    
    return metric_tracker

# Referred to DSINE
def evaluation_normal(eval_dir, base_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                      gen_prediction=None, pipeline=None, prediction_dir=None, processing_res=None,
    eval_datasets=[('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel')],
    save_pred_vis=False
    ):
    '''
    if eval_mode == "load_prediction": assert prediction_dir is not None
    elif eval_mode == "generate_prediction": assert gen_prediction is not None and pipeline is not None
    '''
    os.makedirs(eval_dir, exist_ok=True)
    logging.info(f"processing_res: {processing_res}")

    device = torch.device('cuda')
    metric_results = {}
    for dataset_name, split in eval_datasets:
        loader = TestLoader(base_data_dir, dataset_split_path, dataset_name_test=dataset_name, test_split=split)
        test_loader = loader.data
        
        results_dir = None
        total_normal_errors = None

        output_dir = os.path.join(eval_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        if save_pred_vis:
            results_dir = os.path.join(output_dir, "vis")
            os.makedirs(results_dir, exist_ok=True)
            print(f"Saving visualizations to {results_dir}")

        for data_dict in tqdm(test_loader):
            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            scene_names = data_dict['scene_name']
            img_names = data_dict['img_name']
            intrins = data_dict['intrins'].to(device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = normal_utils.get_padding(orig_H, orig_W)
            img, intrins = normal_utils.pad_input(img, intrins, lrtb)

            # forward pass
            # pred_list = model(img, intrins=intrins, mode='test')
            # norm_out = pred_list[-1] # [1, 3, h, w]
            if eval_mode == "load_prediction":
                pred_path = os.path.join(prediction_dir, dataset_name, f'{scene_names[0]}_{img_names[0]}_norm.png')
                norm_out = cv2.cvtColor(cv2.imread(pred_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                norm_out = (norm_out.astype(np.float32) / 255.0) * 2.0 - 1.0 # np.array([h,w,3])
                norm_out = torch.tensor(norm_out).permute(2,0,1).unsqueeze(0).to(device) # torch.tensor([1, 3, h, w])

            elif eval_mode == "generate_prediction":
                # resize to processing_res
                if processing_res is not None:
                    input_size = img.shape
                    img =  resize_max_res(
                    img, max_edge_resolution=processing_res,
                    # resample_method=resample_method,
                    )
                norm_out = gen_prediction(img, pipeline) # [1, 3, h, w]
                
                # resize to original res
                if processing_res is not None:
                    norm_out = resize(norm_out, input_size[-2:], antialias=True, )

            # crop the padded part
            norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
            #↑↑↑↑

            if 'normal' in data_dict.keys():
                gt_norm = data_dict['normal'].to(device)
                gt_norm_mask = data_dict['normal_mask'].to(device)
                # # resize gt_norm to original size
                # pred_norm = resize(pred_norm, (gt_norm.shape[-2], gt_norm.shape[-1]), antialias=True)
                # # import torchvision; torchvision.utils.save_image(pred_norm, 'pred_norm.png')
                # # import torchvision; torchvision.utils.save_image(gt_norm, 'gt_norm.png')
                # # import torchvision; torchvision.utils.save_image(gt_norm_mask.float(), 'gt_norm_mask.png')
                # # breakpoint()

                pred_error = normal_utils.compute_cosine_error(pred_norm, gt_norm)
                if total_normal_errors is None:
                    total_normal_errors = pred_error[gt_norm_mask]
                else:
                    total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            if results_dir is not None:
                prefixs = ['%s_%s' % (i,j) for (i,j) in zip(scene_names, img_names)]
                vis_utils.visualize_normal(results_dir, prefixs, img, pred_norm, pred_kappa,
                                        gt_norm, gt_norm_mask, pred_error)

        metrics = None
        if total_normal_errors is not None:
            metrics = normal_utils.compute_normal_metrics(total_normal_errors)
            print("Dataset: ", dataset_name)
            print("mean median rmse 5 7.5 11.25 22.5 30")
            print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

        metric_results[dataset_name] = metrics
        # -------------------- Save metrics to file --------------------
        eval_text = f"Evaluation metrics:\n\
        on dataset: {dataset_name}\n\
        with samples in: {loader.test_samples.split_path}\n"

        eval_text += tabulate(
        [metrics.keys(), metrics.values()]
        )

        _save_to = os.path.join(output_dir, "eval_metrics.txt")
        with open(_save_to, "w+") as f:
            f.write(eval_text)
            logging.info(f"Evaluation metrics saved to {_save_to}")

        
    return metric_results


def evaluation_material(
        eval_dir, base_data_dir, dataset_split_path, eval_mode="generate_prediction", 
        gen_prediction=None, pipeline=None, accelerator=None, generator=None, 
        prediction_dir=None, processing_res=None,
        eval_datasets=[('lightstage', 'test')],
        save_pred_vis=False, args=None, task='curr', model_alias='model', inverse_model_alias=''
    ):
    '''
    if eval_mode == "load_prediction": assert prediction_dir is not None
    elif eval_mode == "generate_prediction": assert gen_prediction is not None and pipeline is not None
    '''
    os.makedirs(eval_dir, exist_ok=True)
    logging.info(f"processing_res: {processing_res}")

    # device = torch.device('cuda')
    metric_results = {}
    curr_task = task
    
    # distributed inference
    distributed_state = PartialState()
    if 'dsine' in model_alias:
        pipeline.model.to(distributed_state.device)
    else:
        pipeline.to(distributed_state.device)

    for dataset_name, split in eval_datasets:
        
        results_dir = None
        output_dir = os.path.join(eval_dir, dataset_name, model_alias)
        if inverse_model_alias:
            output_dir = os.path.join(eval_dir, dataset_name, f'{model_alias}_via_{inverse_model_alias}')
        inverse_output_dir = os.path.join(os.path.dirname(eval_dir), f'step00000', dataset_name, inverse_model_alias) if inverse_model_alias else None
        os.makedirs(output_dir, exist_ok=True)
        
        if not inverse_model_alias or inverse_model_alias == 'gt':
            pass
        else:
            if not os.path.exists(inverse_output_dir):
                print(f'Inverse output dir {inverse_output_dir} does not exist, skip.')
                continue

        if dataset_name == 'lightstage':
            
            # check if already exists
            metrics_out_path = os.path.join(output_dir, f"{curr_task}_metrics.txt")
            if os.path.exists(metrics_out_path):
                print(f"Metrics file {metrics_out_path} already exists, skip evaluation.")
                continue
            
            from utils.lightstage_dataset import LightstageDataset, collate_fn_lightstage
            test_dataset_lightstage = LightstageDataset(split=split, tasks=curr_task, ori_aug_ratio=args.lightstage_original_augmentation_ratio, lighting_aug=args.lightstage_lighting_augmentation, eval_first_n_item=args.evaluation_top_k)
        # print(f'Number of samples in {dataset_name} {split}: {len(samples)}')
        bsz = 1
        test_loader = DataLoader(
            test_dataset_lightstage, 
            batch_size=bsz, 
            shuffle=False, 
            collate_fn=collate_fn_lightstage,
            num_workers=1, 
            pin_memory=True
        )
        test_loader = accelerator.prepare(test_loader)

        if save_pred_vis:
            results_dir = os.path.join(output_dir, "vis")
            os.makedirs(results_dir, exist_ok=True)
            print(f"Saving visualizations to {results_dir}/{curr_task}")
            
        def get_normal():
            
            total_normal_errors = None
            if eval_mode == "load_prediction":
                pred_path = os.path.join(prediction_dir, dataset_name, f'{img_meta["obj"]}_norm.png')
                norm_out = cv2.cvtColor(cv2.imread(pred_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                norm_out = (norm_out.astype(np.float32) / 255.0) * 2.0 - 1.0 # np.array([h,w,3])
                norm_out = torch.tensor(norm_out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            elif eval_mode == "generate_prediction":
                if 'rgb2x' in model_alias:
                    img_ret, pred_ret, prompts_ret = gen_prediction(img_path, pipeline, accelerator, generator, img_rgb=img, required_aovs=['normal']) # call rgb2x
                    norm_out = pred_ret[list(prompts_ret.keys()).index('normal')][0] # PIL
                    norm_out = (np.asarray(norm_out).astype(np.float32) / 255.0) * 2.0 - 1.0 # np.array([h,w,3]), [-1,1]
                    norm_out = torch.tensor(norm_out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])
                elif 'lotus' in model_alias:
                    norm_out = gen_prediction(img_path, pipeline, accelerator) # call rgb2x # [1, 3, h, w], [-1,1]
                elif 'dsine' in model_alias:
                    # Use the model to infer the normal map from the input image
                    with torch.inference_mode():
                        normal = pipeline.infer_cv2((img*255.).to(torch.float32).permute(1, 2, 0).cpu().numpy()) # call dsine normal predictor, Output shape: (3, H, W), [-1,1]
                    norm_out = normal # [1, 3, h, w], [-1,1]

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            if 'normal_values' in data_dict.keys():
                gt_norm = data_dict['normal_values'].to(distributed_state.device)
                gt_norm_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt_norm[:, :1, :, :], dtype=torch.bool)

                pred_error = normal_utils.compute_cosine_error(pred_norm, gt_norm) # [B, C, H, W]
                # if total_normal_errors is None:
                #     total_normal_errors = pred_error[gt_norm_mask]
                # else:
                #     total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)
                metrics = normal_utils.cross_verify_metrics(pred_norm*0.5+0.5, gt_norm*0.5+0.5, mask=gt_norm_mask.repeat(1,3,1,1))

                if results_dir is not None:
                    if 'i' in img_meta:
                        prefixs = [f'{o}_cam{c}_l{l}_i{i}_j{j}' for (o,c,l,i,j) in zip(img_meta['obj'], img_meta['cam'], img_meta['l'], img_meta['i'], img_meta['j'])]
                    else:
                        prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                    mosaic = vis_utils.visualize_normal(os.path.join(results_dir, curr_task, 'separated'), prefixs, img[None,...], pred_norm, pred_kappa, gt_norm, gt_norm_mask, pred_error)

                return mosaic, metrics

        def get_albedo():
            if eval_mode == "generate_prediction":
                if 'rgb2x' in model_alias:
                    img_ret, pred_ret, prompts_ret = gen_prediction(img_path, pipeline, accelerator, generator, img_rgb=img, required_aovs=['albedo']) # call rgb2x via img, for jpg input, results input checked the same
                    albedo_out = pred_ret[list(prompts_ret.keys()).index('albedo')][0] # PIL
                    albedo_out = (np.asarray(albedo_out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                    albedo_out = torch.tensor(albedo_out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            pred_albedo, pred_kappa = albedo_out[:, :3, :, :], albedo_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            # save prediction and gt
            if 'albedo_values' in data_dict.keys():
                gt_albedo = (data_dict['albedo_values'].to(distributed_state.device) + 1.0) / 2.0 # albedo_values is scaled to [-1,1] in dataloader, recover here
                gt_albedo_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt_albedo[:, :1, :, :], dtype=torch.bool)
                pred_error = normal_utils.compute_cosine_error(pred_albedo, gt_albedo)
                metrics = normal_utils.cross_verify_metrics(pred_albedo*0.5+0.5, gt_albedo*0.5+0.5, mask=gt_albedo_mask.repeat(1,3,1,1))

                # save visualization mosaic
                if results_dir is not None:
                    if 'i' in img_meta:
                        prefixs = [f'{o}_cam{c}_l{l}_i{i}_j{j}' for (o,c,l,i,j) in zip(img_meta['obj'], img_meta['cam'], img_meta['l'], img_meta['i'], img_meta['j'])]
                    else:
                        prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                    mosaic = vis_utils.visualize_albedo(os.path.join(results_dir, curr_task, 'separated'), prefixs, img[None,...], pred_albedo, pred_kappa, gt_albedo, gt_albedo_mask, pred_error)
                
                return mosaic, metrics
        def get_specular():
            if eval_mode == "generate_prediction":
                if 'rgb2x' in model_alias:
                    img_ret, pred_ret, prompts_ret = gen_prediction(img_path, pipeline, accelerator, generator, img_rgb=img, required_aovs=['specular']) # call rgb2x via img, for jpg input, results input checked the same
                    out = pred_ret[list(prompts_ret.keys()).index('specular')][0] # PIL
                    out = (np.asarray(out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                    out = torch.tensor(out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            pred, pred_kappa = out[:, :3, :, :], out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            # save prediction and gt
            if 'specular_values' in data_dict.keys():
                gt = (data_dict['specular_values'].to(distributed_state.device) + 1.0) / 2.0 # specular_values is scaled to [-1,1] in dataloader, recover here
                gt_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt[:, :1, :, :], dtype=torch.bool)
                pred_error = normal_utils.compute_cosine_error(pred, gt)
                metrics = normal_utils.cross_verify_metrics(pred*0.5+0.5, gt*0.5+0.5, mask=gt_mask.repeat(1,3,1,1))
                if results_dir is not None:
                    prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                    mosaic = vis_utils.visualize_albedo(os.path.join(results_dir, curr_task, 'separated'), prefixs, img[None,...], pred, pred_kappa, gt, gt_mask, pred_error)
                return mosaic, metrics
            
        def get_cross():
            if eval_mode == "generate_prediction":
                if 'rgb2x' in model_alias:
                    img_ret, pred_ret, prompts_ret = gen_prediction(img_path, pipeline, accelerator, generator, img_rgb=img, required_aovs=['cross']) # call rgb2x via img, for jpg input, results input checked the same
                    out = pred_ret[list(prompts_ret.keys()).index('cross')][0] # PIL
                    out = (np.asarray(out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                    out = torch.tensor(out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            pred, pred_kappa = out[:, :3, :, :], out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            # save prediction and gt
            if 'static_cross_values' in data_dict.keys():
                if pidx == 0:
                    gt = (data_dict['static_cross_values'].to(distributed_state.device) + 1.0) / 2.0 # static_cross_values is scaled to [-1,1] in dataloader, recover here
                else:
                    gt = (data_dict['cross_values'][:,pidx-1].to(distributed_state.device) + 1.0) / 2.0 # cross_values is scaled to [-1,1] in dataloader, recover here
                gt_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt[:, :1, :, :], dtype=torch.bool)
                pred_error = normal_utils.compute_cosine_error(pred, gt)
                metrics = normal_utils.cross_verify_metrics(pred*0.5+0.5, gt*0.5+0.5, mask=gt_mask.repeat(1,3,1,1))
                if results_dir is not None:
                    prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                    mosaic = vis_utils.visualize_albedo(os.path.join(results_dir, curr_task, 'separated'), prefixs, img[None,...], pred, pred_kappa, gt, gt_mask, pred_error)
                return mosaic, metrics
            
        def get_parallel():
            if eval_mode == "generate_prediction":
                if 'rgb2x' in model_alias:
                    img_ret, pred_ret, prompts_ret = gen_prediction(img_path, pipeline, accelerator, generator, img_rgb=img, required_aovs=['parallel']) # call rgb2x via img, for jpg input, results input checked the same
                    out = pred_ret[list(prompts_ret.keys()).index('parallel')][0] # PIL
                    out = (np.asarray(out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                    out = torch.tensor(out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            pred, pred_kappa = out[:, :3, :, :], out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa

            # save prediction and gt
            if 'static_parallel_values' in data_dict.keys():
                if pidx == 0:
                    gt = (data_dict['static_parallel_values'].to(distributed_state.device) + 1.0) / 2.0 # static_parallel_values is scaled to [-1,1] in dataloader, recover here
                else:
                    gt = (data_dict['parallel_values'][:,pidx-1].to(distributed_state.device) + 1.0) / 2.0 # parallel_values is scaled to [-1,1] in dataloader, recover here
                gt_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt[:, :1, :, :], dtype=torch.bool)
                pred_error = normal_utils.compute_cosine_error(pred, gt)
                metrics = normal_utils.cross_verify_metrics(pred*0.5+0.5, gt*0.5+0.5, mask=gt_mask.repeat(1,3,1,1))
                if results_dir is not None:
                    prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                    mosaic = vis_utils.visualize_albedo(os.path.join(results_dir, curr_task, 'separated'), prefixs, img[None,...], pred, pred_kappa, gt, gt_mask, pred_error)
                return mosaic, metrics
            
        def get_forward():
            
            if eval_mode == "generate_prediction":
                img_black = torch.zeros_like(img)
                img_white = torch.ones_like(img)
                if 'x2rgb' in model_alias:
                    img_albedo = data_dict['albedo_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    img_normal = data_dict['normal_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    img_specular = data_dict['specular_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    img_static_cross = data_dict['static_cross_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    img_static_parallel = data_dict['static_parallel_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    # img_normal = data_dict['normal_c2w_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                    img_irradiance = img_white if pidx == 0 else data_dict['irradiance_values'][bsz-1][pidx-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1], only use irradiance for augmented images
                    # img_irradiance = data_dict['pixel_irradiance_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1], only use irradiance for augmented images

                    if not inverse_model_alias or inverse_model_alias == 'gt':
                        if curr_task == 'forward_gbuffer':
                            img_ret, pred_ret, prompts_ret = gen_prediction('', '', '', '', '', '', '', pipeline, accelerator, generator, img_rgb=img, img_albedo=img_albedo, img_normal=img_normal, img_roughness=img_specular, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                        elif curr_task == 'forward_gbuffer_albedo_only':
                            img_ret, pred_ret, prompts_ret = gen_prediction('', '', '', '', '', '', '', pipeline, accelerator, generator, img_rgb=img, img_albedo=img_albedo, img_normal=img_black, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                        elif curr_task == 'forward_polarization':
                            img_ret, pred_ret, prompts_ret = gen_prediction('', '', '', '', '', '', '', pipeline, accelerator, generator, img_rgb=img, img_albedo=img_static_cross, img_normal=img_static_parallel, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                    else:
                        prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                        
                        
                        if curr_task == 'forward_gbuffer':
                            img_ret, pred_ret, prompts_ret = gen_prediction(
                                '', 
                                os.path.join(inverse_output_dir, 'vis', 'albedo', 'separated', prefixs[0], 'pred_albedo.png'), 
                                os.path.join(inverse_output_dir, 'vis', 'normal', 'separated', prefixs[0], 'norm.png'), 
                                os.path.join(inverse_output_dir, 'vis', 'specular', 'separated', prefixs[0], 'pred_albedo.png'), 
                                '', 
                                '', # irradiance
                                '', pipeline, accelerator, generator, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=None, img_metallic=img_black, img_irradiance=img_irradiance
                            ) # call rgb2x via img, for jpg input, results input checked the same
                        elif curr_task == 'forward_gbuffer_albedo_only':
                            img_ret, pred_ret, prompts_ret = gen_prediction(
                                '', 
                                os.path.join(inverse_output_dir, 'vis', 'albedo', 'separated', prefixs[0], 'pred_albedo.png'), 
                                '', 
                                '', 
                                '', 
                                '', # irradiance
                                '', pipeline, accelerator, generator, img_rgb=img, img_albedo=None, img_normal=img_black, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance
                            ) # call rgb2x via img, for jpg input, results input checked the same
                        elif curr_task == 'forward_polarization':
                            img_ret, pred_ret, prompts_ret = gen_prediction(
                                '', 
                                os.path.join(inverse_output_dir, 'vis', 'cross', 'separated', prefixs[0], 'pred_albedo.png'), 
                                os.path.join(inverse_output_dir, 'vis', 'parallel', 'separated', prefixs[0], 'pred_albedo.png'), 
                                '', 
                                '', 
                                '', # irradiance
                                '', pipeline, accelerator, generator, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance
                            ) # call rgb2x via img, for jpg input, results input checked the same
                        
                    img_out = pred_ret[bsz-1][0] # PIL
                    img_out = (np.asarray(img_out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                    img_out = torch.tensor(img_out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])

            pred_img, pred_kappa = img_out[:, :3, :, :], img_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
            
            # save prediction and gt
            if 'static_values' in data_dict.keys():
                gt_img = img[None,...] # img in [0,1]
                gt_pixel_mask = data_dict['valid_mask_values'].to(distributed_state.device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt_img[:, :1, :, :], dtype=torch.bool)

                pred_error = normal_utils.compute_cosine_error(pred_img, gt_img)
                # if total_normal_errors is None:
                #     total_normal_errors = pred_error[gt_pixel_mask]
                # else:
                #     total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_pixel_mask]), dim=0)
                metrics = normal_utils.cross_verify_metrics(pred_img*0.5+0.5, gt_img*0.5+0.5, mask=gt_pixel_mask.repeat(1,3,1,1))

            # save visualization mosaic
            if results_dir is not None:
                in_irradiance = img_irradiance[None,...]
                in_albedo = img_albedo[None,...]
                if 'i' in img_meta:
                    prefixs = [f'{o}_cam{c}_l{l}_i{i}_j{j}' for (o,c,l,i,j) in zip(img_meta['obj'], img_meta['cam'], img_meta['l'], img_meta['i'], img_meta['j'])]
                else:
                    prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                mosaic = vis_utils.visualize_img(os.path.join(results_dir, curr_task, 'separated'), prefixs, gt_img, in_albedo, in_irradiance, pred_img, pred_kappa, pred_error)
                # mosaics.append(mosaic)
                
                # rendering olat
                step = eval_dir.split('/')[-1].replace('step','')
                if int(step) % args.evaluation_olat_steps == 0 and pidx == 0:
                    import imageio
                
                    results_dir_olat = os.path.join(output_dir, 'olat')
                    os.makedirs(results_dir_olat, exist_ok=True)
                    results_dir_olat_frames = os.path.join(results_dir_olat, 'frames', f'{img_meta["obj"]}_cam{img_meta["cam"]}')
                    os.makedirs(results_dir_olat_frames, exist_ok=True)
                    
                    olat_target_frames = 20
                    N_OLATS = test_dataset_lightstage.omega_i_world.shape[0]
                    intensity_scalar = test_dataset_lightstage.olat_wi_rgb_intensity['random1'] / N_OLATS
                    olat_img_scalar = test_dataset_lightstage.olat_img_intensity['random1']
                    olat_step = N_OLATS // olat_target_frames
                    
                    Ls = torch.tensor(test_dataset_lightstage.omega_i_world[::olat_step]).to(img.device) # (n, 3)
                    nDotL = torch.einsum('nc,chw->nhw', Ls, data_dict['normal_ls_c2w_values'][bsz-1]) # (n, h, w)
                    L_rgb = intensity_scalar * torch.ones_like(Ls) # (n, 3)
                    L_irradiance = torch.einsum('nhw,nc->nchw', torch.maximum(nDotL, torch.tensor(0.0)), L_rgb) # (3, h, w), align with random 1
                    frame_mosaics = []
                    for i, img_irradiance_ in enumerate(L_irradiance[:olat_target_frames//2]): # (3, h, w)
                        # render one view
                        img_ret, pred_ret, prompts_ret = gen_prediction('', '', '', '', '', '', '', pipeline, accelerator, generator, img_rgb=img, img_albedo=img_albedo, img_normal=img_black, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance_) # call rgb2x via img, for jpg input, results input checked the same
                        frmae_out = pred_ret[bsz-1][0] # PIL
                        frmae_out = (np.asarray(frmae_out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                        frmae_out = torch.tensor(frmae_out).permute(2,0,1).unsqueeze(0).to(distributed_state.device) # torch.tensor([1, 3, h, w])
                        frmae_out = frmae_out[:, :3, :, :]
                        frmae_out = frmae_out.clamp(0,1)
                        
                        frame_id = i*olat_step
                        # save frame
                        in_irradiance_ = img_irradiance_[None,...]
                        gt_img_ = (olat_img_scalar * intensity_scalar * torch.tensor(imageio.imread(os.path.join(os.path.dirname(data_dict['parallel_paths'][bsz-1][0][0]), f'{frame_id+2:06d}.jpg')) / 255.0).permute(2,0,1).unsqueeze(0).to(distributed_state.device)).clip(0, 1) # [1, 3, h, w], load gt from the lightstage video folder
                        prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [i])]
                        frame_mosaic = vis_utils.visualize_img(results_dir_olat_frames, prefixs, gt_img_, in_albedo, in_irradiance_, frmae_out, pred_kappa, pred_error)
                        frame_mosaics.append(frame_mosaic)
                        
                    # save a mp4 video
                    imageio.mimwrite(os.path.join(results_dir_olat, f'{prefixs[0]}_olat.mp4'), frame_mosaics, fps=5, quality=8)
                    
                return mosaic, metrics
            
        ##################### Evaluate #####################
        metrics_dataset = []
        with distributed_state.split_between_processes(test_loader) as validation_batch:
            for data_dict in tqdm(validation_batch, desc=f"Evaluating {dataset_name} {split} via {model_alias} + {inverse_model_alias} on {distributed_state.device}"):
                #↓↓↓↓
                #NOTE: forward pass
                img_path = data_dict['static_paths'][bsz-1] # e.g. 'lightstage/scene1/img1.png'
                img_meta = data_dict['metas'][bsz-1]
                intrins = NotImplemented
                
                # also evaluate the image pairs
                img_pairs = [data_dict['static_values'].to(distributed_state.device)[bsz-1]] # static image first
                img_pairs += [parallel_img.to(distributed_state.device) for parallel_img in data_dict['parallel_values'][bsz-1]] # and then the parallel images

                mosaics = []
                for pidx, img_ in enumerate(img_pairs):
                    # forward pass
                    img = (img_ * 0.5 + 0.5).clamp(0, 1) # img_ is in [-1,1]

                    if curr_task == 'normal':
                        mosaic, metrics = get_normal()
                    elif curr_task == 'albedo':
                        mosaic, metrics = get_albedo()
                    elif curr_task == 'specular':
                        mosaic, metrics = get_specular()
                    elif curr_task == 'cross':
                        mosaic, metrics = get_cross()
                    elif curr_task == 'parallel':
                        mosaic, metrics = get_parallel()
                    elif curr_task == 'forward_gbuffer':
                        mosaic, metrics = get_forward()
                    elif curr_task == 'forward_gbuffer_albedo_only':
                        mosaic, metrics = get_forward()
                    elif curr_task == 'forward_polarization':
                        mosaic, metrics = get_forward()

                    if mosaic is not None:
                        mosaics.append(mosaic)
                        metrics_dataset.append(metrics)

                # concatnate mosaics
                if len(mosaics) > 0:
                    mosaics = np.concatenate(mosaics, axis=0)  # (H*3, W, 3)
                    target_path = '%s/%s.png' % (os.path.join(results_dir, curr_task, 'mosaics'), f'{img_meta["obj"]}_cam{img_meta["cam"]}_l{img_meta["l"]}_all')
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    plt.imsave(target_path, mosaics)

        metrics_mse = [m['mse'].get('torchmetrics', 0) for m in metrics_dataset]
        metrics_psnr = [m['psnr'].get('torchmetrics', 0) for m in metrics_dataset]
        metrics_ssim = [m['ssim'].get('torchmetrics', 0) for m in metrics_dataset]
        metrics_lpips = [m['lpips'].get('torchmetrics', 0) for m in metrics_dataset]

        metrics_mse_static = [m['mse'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) == 0]
        metrics_psnr_static = [m['psnr'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) == 0]
        metrics_ssim_static = [m['ssim'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) == 0]
        metrics_lpips_static = [m['lpips'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) == 0]

        metrics_mse_olat_aug = [m['mse'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) != 0]
        metrics_psnr_olat_aug = [m['psnr'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) != 0]
        metrics_ssim_olat_aug = [m['ssim'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) != 0]
        metrics_lpips_olat_aug = [m['lpips'].get('torchmetrics', 0) for i, m in enumerate(metrics_dataset) if i % (1 + len(data_dict['parallel_values'][bsz-1])) != 0]

        # write the metrics summary table to file
        metrics_out_path = os.path.join(output_dir, f"{curr_task}_metrics.txt")
        with open(metrics_out_path, "w+") as f:
            eval_text = f"Evaluation metrics:\n\
            of predictions: {prediction_dir}\n\
            on dataset: {dataset_name}\n\
            with samples in: {dataset_split_path}\n"
            eval_text += f"Number of samples: {len(metrics_dataset)}\n"
            eval_text += tabulate(
                [
                    
                    ['mse', 'psnr', 'ssim', 'lpips'],
                    [f"{np.mean(metrics_mse):.6f} ± {np.std(metrics_mse):.6f}", 
                     f"{np.mean(metrics_psnr):.3f} ± {np.std(metrics_psnr):.3f}", 
                     f"{np.mean(metrics_ssim):.3f} ± {np.std(metrics_ssim):.3f}", 
                     f"{np.mean(metrics_lpips):.3f} ± {np.std(metrics_lpips):.3f}"]
                ]
            )
            eval_text += f"\nNumber of static samples: {len(metrics_mse_static)}\n"
            eval_text += tabulate(
                [
                    
                    ['mse', 'psnr', 'ssim', 'lpips'],
                    [f"{np.mean(metrics_mse_static):.6f} ± {np.std(metrics_mse_static):.6f}", 
                     f"{np.mean(metrics_psnr_static):.3f} ± {np.std(metrics_psnr_static):.3f}", 
                     f"{np.mean(metrics_ssim_static):.3f} ± {np.std(metrics_ssim_static):.3f}", 
                     f"{np.mean(metrics_lpips_static):.3f} ± {np.std(metrics_lpips_static):.3f}"]
                ]
            )
            eval_text += f"\nNumber of olat augmented samples: {len(metrics_mse_olat_aug)}\n"
            eval_text += tabulate(
                [
                    ['mse', 'psnr', 'ssim', 'lpips'],
                    [f"{np.mean(metrics_mse_olat_aug):.6f} ± {np.std(metrics_mse_olat_aug):.6f}", 
                     f"{np.mean(metrics_psnr_olat_aug):.3f} ± {np.std(metrics_psnr_olat_aug):.3f}", 
                     f"{np.mean(metrics_ssim_olat_aug):.3f} ± {np.std(metrics_ssim_olat_aug):.3f}", 
                     f"{np.mean(metrics_lpips_olat_aug):.3f} ± {np.std(metrics_lpips_olat_aug):.3f}"]
                ]
            )
            
            f.write(eval_text)
            logging.info(f"Evaluation metrics saved to {metrics_out_path}")

        # metrics = None
        # if total_normal_errors is not None:
        #     metrics = normal_utils.compute_normal_metrics(total_normal_errors)
        #     print("Dataset: ", dataset_name)
        #     print("mean median rmse 5 7.5 11.25 22.5 30")
        #     print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        #         metrics['mean'], metrics['median'], metrics['rmse'],
        #         metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

        metric_results[dataset_name] = metrics_dataset

        
    return metric_results