import sys
import PIL
import numpy as np
import os
import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
import imageio

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../lotus')))
from lotus.utils.lightstage_dataset import LightstageDataset, collate_fn_lightstage
from lotus.utils.objaverse_dataset import ObjaverseDataset, collate_fn_objaverse
import lotus.utils.visualize as vis_utils
from lotus.train_lotus_g_rgb2x import rgb2x
from lotus.train_lotus_g_rgb2x import x2rgb
from lotus.evaluation.util import normal_utils
from lotus.pipeline import LotusGPipeline, LotusDPipeline

from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from rgbx.x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

if __name__ == "__main__":
    
    first_n = 2 if len(sys.argv) < 2 else int(sys.argv[1])
    print(f"Evaluating first {first_n} samples from lightstage dataset")
    
    # datasets = ['lightstage', 'objaverse', 'multi_illumination', 'interiorverse']
    datasets = ['lightstage', 'objaverse']
    # datasets = ['objaverse']
    # datasets = ['lightstage']
    
    
    outdir = 'output/evaluation'
    exp_name = 'step00000_frontal_irradiance'
    bsz = 1
    
                
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
                
    for dataset_name in datasets:
        
        # inverse 
        # pipelines_name = ['rgb2x_pretrain', 'rgb2x_posttrain', 'lotus', '']
        # pipelines_name = ['rgb2x_pretrain']
        # pipelines_name = ['rgb2x_posttrain']
        # forward
        # pipelines_name = ['x2rgb_pretrain_via_rgb2x_pretrain', 'x2rgb_posttrain_via_rgb2x_posttrain', 'x2rgb_posttrain_via_gt']
        # pipelines_name = ['x2rgb_pretrain_via_rgb2x_pretrain']
        
        if dataset_name == 'lightstage':
            pipelines_name = ['rgb2x_pretrain', 'rgb2x_posttrain', 'x2rgb_pretrain_via_rgb2x_pretrain', 'x2rgb_posttrain_via_rgb2x_posttrain', 'x2rgb_posttrain_via_gt']
            # pipelines_name = ['x2rgb_pretrain_via_rgb2x_pretrain', 'x2rgb_posttrain_via_rgb2x_posttrain', 'x2rgb_posttrain_via_gt']
        elif dataset_name == 'objaverse':
            pipelines_name = ['rgb2x_pretrain', 'rgb2x_posttrain', 'x2rgb_pretrain_via_rgb2x_pretrain', 'x2rgb_posttrain_via_rgb2x_posttrain', 'x2rgb_posttrain_via_gt']
            # pipelines_name = ['x2rgb_pretrain_via_rgb2x_pretrain', 'x2rgb_posttrain_via_rgb2x_posttrain', 'x2rgb_posttrain_via_gt']

        # pipelines_name = ['lotus-normal-g-v1-1', 'lotus-normal-d-v1-1', 'dsine']

        for pipeline_name in pipelines_name:
            
            if dataset_name == 'lightstage':
                if 'rgb2x_' in pipeline_name.split("_via_")[0]:
                    tasks = ['albedo', 'normal', 'specular', 'cross', 'parallel']  # lightstage only evaluate albedo for now
                    # tasks = ['cross']
                    # tasks = ['parallel']  # lightstage only evaluate albedo for now
                elif 'x2rgb_' in pipeline_name.split("_via_")[0]:
                    tasks = ['forward_gbuffer', 'forward_polarization']
                elif 'lotus-normal' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']
                elif 'dsine' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']
                elif 'controlnet-normal' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']
            elif dataset_name == 'objaverse':
                if 'rgb2x_' in pipeline_name.split("_via_")[0]:
                    tasks = ['albedo', 'normal', 'specular', 'cross', 'parallel']
                    # tasks = ['cross', 'parallel']  # lightstage only evaluate albedo for now
                elif 'x2rgb_' in pipeline_name.split("_via_")[0]:
                    tasks = ['forward_gbuffer', 'forward_polarization']
                elif 'lotus-normal' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']
                elif 'dsine' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']
                elif 'controlnet-normal' in pipeline_name.split("_via_")[0]:
                    tasks = ['normal']

            for task in tasks:
                
                # clean up cache
                torch.cuda.empty_cache()
                print(f"Evaluating dataset: {dataset_name}, pipeline: {pipeline_name}, task: {task}")
                
                # load pipeline
                if pipeline_name.split("_via_")[0] == 'rgb2x_pretrain':
                    pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained('zheng95z/rgb-to-x')
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif pipeline_name.split("_via_")[0] == 'rgb2x_posttrain':
                    # this section requires lotus env therefor eneed to be 
                    pretrained_inverse_model_path = './output/benchmark/train-rgb2x-lora-inverse-bsz32/rgb2x-lora-inverse-ckpt4fr'
                    ckpts = [dirname for dirname in os.listdir(pretrained_inverse_model_path) if 'checkpoint-' in dirname]
                    ckpts.sort(key=lambda x: int(x.split('checkpoint-')[-1]), reverse=True)
                    assert len(ckpts) > 0, "No checkpoints found"
                    pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained('zheng95z/rgb-to-x')
                    pipeline.load_lora_weights(os.path.join(pretrained_inverse_model_path, ckpts[0]))
                    assert pipeline.get_active_adapters(), "LoRA weights not loaded properly"
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif pipeline_name.split("_via_")[0] == 'x2rgb_pretrain':
                    pipeline = StableDiffusionAOVDropoutPipeline.from_pretrained('zheng95z/x-to-rgb')
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif pipeline_name.split("_via_")[0] == 'x2rgb_posttrain':
                    if task == 'forward_gbuffer':
                        pretrained_forward_model_path = './output/benchmark/train-x2rgb-lora-forward_gbuffer-bsz32/saved_ckpt'
                    elif task == 'forward_polarization':
                        pretrained_forward_model_path = './output/benchmark/train-x2rgb-lora-forward_polarization-bsz32/saved_ckpt'
                    ckpts = [dirname for dirname in os.listdir(pretrained_forward_model_path) if 'checkpoint-' in dirname]
                    ckpts.sort(key=lambda x: int(x.split('checkpoint-')[-1]), reverse=True)
                    assert len(ckpts) > 0, "No checkpoints found"
                    pipeline = StableDiffusionAOVDropoutPipeline.from_pretrained('zheng95z/x-to-rgb')
                    pipeline.load_lora_weights(os.path.join(pretrained_forward_model_path, ckpts[0]))
                    assert pipeline.get_active_adapters(), "LoRA weights not loaded properly"
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif 'lotus-normal-g' in pipeline_name.split("_via_")[0]:
                    pipeline = LotusGPipeline.from_pretrained('jingheya/lotus-normal-g-v1-1')
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif 'lotus-normal-d' in pipeline_name.split("_via_")[0]:
                    pipeline = LotusDPipeline.from_pretrained('jingheya/lotus-normal-d-v1-1')
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                elif 'dsine' in pipeline_name.split("_via_")[0]:
                    pipeline = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
                elif 'controlnet-normal' in pipeline_name.split("_via_")[0]:
                    controlnet = ControlNetModel.from_pretrained(
                        "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
                    )
                    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
                    )
                    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline.to(device)
                    pipeline.set_progress_bar_config(disable=True)
                else:
                    raise NotImplementedError(f"Pipeline {pipeline_name} not implemented")
                
                # prepare dataset and dataloader
                if dataset_name == 'lightstage':

                    test_dataset_lightstage = LightstageDataset(split='test', tasks='', ori_aug_ratio="1:1", lighting_aug='fixed20_via1', eval_first_n=50)
                    sampler = DistributedSampler(test_dataset_lightstage, num_replicas=world_size, rank=rank, shuffle=False)
                    dataloader = DataLoader(
                        test_dataset_lightstage, 
                        batch_size=bsz, 
                        shuffle=False, 
                        sampler=sampler,
                        collate_fn=collate_fn_lightstage,
                        num_workers=1, 
                        pin_memory=True
                    )
                    
                    # loop through lightstage dataset
                    n_test = len(dataloader)
                    iter_dataloader = iter(dataloader)
                    metrics_dataset = []
                    for i in range(n_test):
                        data_dict = next(iter_dataloader)
                        obj_id = data_dict['objs'][bsz-1]
                        img_meta = data_dict['metas'][bsz-1]
                        
                        # if obj_id not in ['rubberredball']:
                        #     continue
                        
                        # also evaluate the image pairs
                        img_pairs = [data_dict['static_values'].to(device)[bsz-1]] # static image first
                        img_pairs += [parallel_img.to(device) for parallel_img in data_dict['parallel_values'][bsz-1]] # and then the parallel images
                        pattern = 'static_parallel'

                        mosaics = []
                        for pidx, img_ in enumerate(img_pairs):
                            
                            img = (img_ * 0.5 + 0.5).clamp(0, 1) # img_ is in [-1,1]
                        
                            path = data_dict[f'{pattern}_paths'][bsz-1]
                            path_base = os.path.dirname(os.path.dirname(path))
                            path_save_out = os.path.join(outdir, exp_name, dataset_name, pipeline_name, obj_id)
                            
                            # check if all 8 image are exist, if not, move to next iter
                            all_file_exist = all([os.path.exists(f'{path_base}/cam0{i}/{pattern}.jpg') for i in range(8)])
                            if not all_file_exist:
                                print(f"Some files do not exist for {obj_id}, skipping...")
                                continue

                            for i in range(1):
                                prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                                results_dir = os.path.join(path_save_out, f'{task}', 'separated')
                                
                                
                                img_black = torch.zeros_like(img)
                                img_white = torch.ones_like(img)
                                if 'rgb2x_' in pipeline_name.split('_via_')[0]:
                                    img_ret, pred_ret, prompts_ret = rgb2x(f'{path_base}/cam0{i}/{pattern}.jpg', pipeline, None, None, img_rgb=img, required_aovs=[task]) # call rgb2x via img, for jpg input, results input checked the same
                                    out = pred_ret[list(prompts_ret.keys()).index(task)][0] # PIL
                                elif 'x2rgb_' in pipeline_name.split('_via_')[0]:
                                    assert '_via_' in pipeline_name, "x2rgb pipeline must specify via which inverse model"
                                    inverse_model_name = pipeline_name.split('_via_')[-1]
                                    forward_model_name = pipeline_name.split('_via_')[0]
                                    inverse_model_result = os.path.join(outdir, exp_name, f'{dataset_name}', inverse_model_name, obj_id)
                                    
                                    img_albedo = data_dict['albedo_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_normal = data_dict['normal_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_specular = data_dict['specular_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_static_cross = data_dict['static_cross_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_static_parallel = data_dict['static_parallel_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_irradiance = img_white if pidx == 0 else (data_dict['irradiance_values'][bsz-1][pidx-1] * 0.5 + 0.5) * 2 # [1, 3, h, w], in [0,1], only use irradiance for augmented images
                                    
                                    if inverse_model_name == 'gt':
                                        if task == 'forward_gbuffer':
                                            img_ret, pred_ret, prompts_ret = x2rgb('', '', '', '', '', '', '', pipeline, None, None, img_rgb=img, img_albedo=img_albedo, img_normal=img_normal, img_roughness=img_specular, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                                        elif task == 'forward_polarization':
                                            img_ret, pred_ret, prompts_ret = x2rgb('', '', '', '', '', '', '', pipeline, None, None, img_rgb=img, img_albedo=img_static_cross, img_normal=img_static_parallel, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                                        out = pred_ret[bsz-1][0] # PIL

                                    elif inverse_model_name == 'rgb2x_pretrain' or inverse_model_name == 'rgb2x_posttrain':
                                        if task == 'forward_gbuffer':
                                            img_ret, pred_ret, prompts_ret = x2rgb(
                                                '', 
                                                os.path.join(inverse_model_result, f'albedo/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'normal/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'specular/separated/{prefixs[0]}/pred_albedo.png'), 
                                                '', 
                                                '', # irradiance
                                                '', pipeline, None, None, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=None, img_metallic=img_black, img_irradiance=img_irradiance
                                            ) # call rgb2x via img, for jpg input, results input checked the same
                                        elif task == 'forward_polarization':
                                            img_ret, pred_ret, prompts_ret = x2rgb(
                                                '', 
                                                os.path.join(inverse_model_result, f'cross/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'parallel/separated/{prefixs[0]}/pred_albedo.png'), 
                                                '', 
                                                '', 
                                                '', # irradiance
                                                '', pipeline, None, None, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance
                                            ) # call rgb2x via img, for jpg input, results input checked the same    
                                        out = pred_ret[bsz-1][0] # PIL
                                elif 'lotus-normal' in pipeline_name.split('_via_')[0]:
                                    rgb_in = img_[None,...].to(device)  # [1, 3, h, w]
                                    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
                                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
                                    out = pipeline(rgb_in=rgb_in, task_emb=task_emb, prompt='', num_inference_steps=1, timesteps=[999]).images[0] # PIL
                                elif 'dsine' in pipeline_name.split('_via_')[0]:
                                    out = pipeline.infer_cv2(img.permute(1, 2, 0).cpu().numpy()*255.)[0].permute(1, 2, 0).cpu().numpy() # [-1,1]
                                    out = (out * 0.5 + 0.5) * 255.0

                                out = (np.asarray(out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                                out = torch.tensor(out).permute(2,0,1).unsqueeze(0).to(device) # torch.tensor([1, 3, h, w])
                                pred, pred_kappa = out[:, :3, :, :], out[:, 3:, :, :]
                                pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
                                
                                gt_key = {
                                    'albedo': 'albedo_values',
                                    'normal': 'normal_values',
                                    'specular': 'specular_values',
                                    'cross': 'cross_values' if pidx != 0 else 'static_cross_values',
                                    'parallel': 'parallel_values' if pidx != 0 else 'static_parallel_values'
                                }
                                
                                reference = img[None,...]
                                if 'cross' in task or 'parallel' in task:
                                    if pidx == 0:
                                        gt = (data_dict[gt_key[task]].to(device) + 1.0) / 2.0 # static_parallel_values is scaled to [-1,1] in dataloader, recover here
                                    else:
                                        gt = (data_dict[gt_key[task]][:,pidx-1].to(device) + 1.0) / 2.0 # parallel_values is scaled to [-1,1] in dataloader, recover here
                                elif 'forward_' in task:
                                    reference = img_irradiance[None,...].to(device) # debugging the lighting
                                    gt = img[None,...].to(device) # debugging the lighting
                                else:
                                    gt = (data_dict[gt_key[task]].to(device) + 1.0) / 2.0
                                gt_mask = data_dict['valid_mask_values'].to(device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt[:, :1, :, :], dtype=torch.bool)
                                pred_error = normal_utils.compute_cosine_error(pred, gt)
                                metrics = normal_utils.cross_verify_metrics(pred*0.5+0.5, gt*0.5+0.5, mask=gt_mask.repeat(1,3,1,1))
                                mosaic = vis_utils.visualize_albedo(results_dir, prefixs, reference, pred, pred_kappa, gt, gt_mask, pred_error)
                                
                                mosaics.append(mosaic)
                                metrics_dataset.append(metrics)
                        # concatnate mosaics
                        if len(mosaics) > 0:
                            mosaics = np.concatenate(mosaics, axis=0)  # (H*3, W, 3)
                            target_path = '%s/%s.png' % (os.path.join(path_save_out, task, 'mosaics'), f'{img_meta["obj"]}_cam{img_meta["cam"]}_l{img_meta["l"]}_all')
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            plt.imsave(target_path, mosaics)
                            
                    # aggregate metrics across all processes
                    all_metrics = [None for _ in range(world_size)]
                    dist.all_gather_object(all_metrics, metrics_dataset)
                    if rank == 0:
                        metrics_dataset = sum(all_metrics, [])

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
                        metrics_out_path = os.path.join(os.path.dirname(path_save_out), f"{task}_metrics.txt")
                        with open(metrics_out_path, "w+") as f:
                            eval_text = f"Evaluation metrics:\n\
                            on dataset: {dataset_name}\n"
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
                        
                elif dataset_name == 'objaverse':
                    
                    test_dataset_objaverse = ObjaverseDataset(split='test', tasks='', ori_aug_ratio="1:1", lighting_aug='fixed20_via1', eval_first_n=30)
                    sampler = DistributedSampler(test_dataset_objaverse, num_replicas=world_size, rank=rank, shuffle=False)
                    dataloader = DataLoader(
                        test_dataset_objaverse, 
                        batch_size=bsz, 
                        shuffle=False, 
                        sampler=sampler,
                        collate_fn=collate_fn_objaverse,
                        num_workers=8, 
                        pin_memory=True
                    )
                    # loop through objaverse dataset
                    n_test = len(dataloader)
                    iter_dataloader = iter(dataloader)
                    metrics_dataset = []
                    for i in range(n_test):
                        data_dict = next(iter_dataloader)
                        obj_id = data_dict['objs'][bsz-1]
                        img_meta = data_dict['metas'][bsz-1]

                        # if obj_id not in ['globe', 'table_vase']:
                        #     continue
                        
                        img_pairs = [data_dict['static_values'][bsz-1]] # static image first
                        img_pairs += [parallel_img for parallel_img in data_dict['parallel_values'][bsz-1]] # and then the parallel images
                        
                        mosaics = []
                        for pidx, img_ in enumerate(img_pairs):
                            
                            img = (img_ * 0.5 + 0.5).clamp(0, 1) # img_ is in [-1,1]
                            path_save_out = os.path.join(outdir, exp_name, dataset_name, pipeline_name, obj_id)
                            
                            # loop through camera views
                            for i in range(1):
                                prefixs = [f'{o}_cam{c}_l{l}' + ('' if not m else f'_{m}') + ('' if not pidx else f'_aug{pidx}') for (o, m,c,l) in zip([img_meta['obj']], [img_meta['des']], [img_meta['cam']], [img_meta['l']])]
                                results_dir = os.path.join(path_save_out, f'{task}', 'separated')

                                img_black = torch.zeros_like(img)
                                img_white = torch.ones_like(img)
                                if 'rgb2x_' in pipeline_name.split('_via_')[0]:
                                    img_ret, pred_ret, prompts_ret = rgb2x(data_dict['static_paths'][bsz-1], pipeline, None, None, img_rgb=img, required_aovs=[task]) # call rgb2x via img, for jpg input, results input checked the same
                                    out = pred_ret[list(prompts_ret.keys()).index(task)][0] # PIL
                                elif 'x2rgb_' in pipeline_name.split('_via_')[0]:
                                    assert '_via_' in pipeline_name, "x2rgb pipeline must specify via which inverse model"
                                    inverse_model_name = pipeline_name.split('_via_')[-1]
                                    forward_model_name = pipeline_name.split('_via_')[0]
                                    inverse_model_result = os.path.join(outdir, exp_name, f'{dataset_name}', inverse_model_name, obj_id)
                                    
                                    img_albedo = data_dict['albedo_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_normal = data_dict['normal_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_specular = data_dict['specular_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_static_cross = data_dict['static_cross_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_static_parallel = data_dict['static_parallel_values'][bsz-1] * 0.5 + 0.5 # [1, 3, h, w], in [0,1]
                                    img_irradiance = img_white if pidx == 0 else (data_dict['irradiance_values'][bsz-1][pidx-1] * 0.5 + 0.5) * 2 # [1, 3, h, w], in [0,1], only use irradiance for augmented images
                                    
                                    if inverse_model_name == 'gt':
                                        if task == 'forward_gbuffer':
                                            img_ret, pred_ret, prompts_ret = x2rgb('', '', '', '', '', '', '', pipeline, None, None, img_rgb=img, img_albedo=img_albedo, img_normal=img_normal, img_roughness=img_specular, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                                        elif task == 'forward_polarization':
                                            img_ret, pred_ret, prompts_ret = x2rgb('', '', '', '', '', '', '', pipeline, None, None, img_rgb=img, img_albedo=img_static_cross, img_normal=img_static_parallel, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance) # call rgb2x via img, for jpg input, results input checked the same
                                        out = pred_ret[bsz-1][0] # PIL

                                    elif inverse_model_name == 'rgb2x_pretrain' or inverse_model_name == 'rgb2x_posttrain':
                                        if task == 'forward_gbuffer':
                                            img_ret, pred_ret, prompts_ret = x2rgb(
                                                '', 
                                                os.path.join(inverse_model_result, f'albedo/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'normal/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'specular/separated/{prefixs[0]}/pred_albedo.png'), 
                                                '', 
                                                '', # irradiance
                                                '', pipeline, None, None, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=None, img_metallic=img_black, img_irradiance=img_irradiance
                                            ) # call rgb2x via img, for jpg input, results input checked the same
                                        elif task == 'forward_polarization':
                                            img_ret, pred_ret, prompts_ret = x2rgb(
                                                '', 
                                                os.path.join(inverse_model_result, f'cross/separated/{prefixs[0]}/pred_albedo.png'), 
                                                os.path.join(inverse_model_result, f'parallel/separated/{prefixs[0]}/pred_albedo.png'), 
                                                '', 
                                                '', 
                                                '', # irradiance
                                                '', pipeline, None, None, img_rgb=img, img_albedo=None, img_normal=None, img_roughness=img_black, img_metallic=img_black, img_irradiance=img_irradiance
                                            ) # call rgb2x via img, for jpg input, results input checked the same    
                                        out = pred_ret[bsz-1][0] # PIL
                                elif 'lotus-normal' in pipeline_name.split('_via_')[0]:
                                    rgb_in = img_[None,...].to(device)  # [1, 3, h, w]
                                    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
                                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
                                    out = pipeline(rgb_in=rgb_in, task_emb=task_emb, prompt='', num_inference_steps=1, timesteps=[999]).images[0] # PIL
                                elif 'dsine' in pipeline_name.split('_via_')[0]:
                                    out = pipeline.infer_cv2(img.permute(1, 2, 0).cpu().numpy()*255.)[0].permute(1, 2, 0).cpu().numpy() # [-1,1]
                                    out = (out * 0.5 + 0.5) * 255.0
                                elif 'controlnet-normal' in pipeline_name.split('_via_')[0]:
                                    out = pipeline('', img[None,...]*255., num_inference_steps=20).images[0]  # PIL
                                        
                                out = (np.asarray(out).astype(np.float32) / 255.0) # np.array([h,w,3]), [0,1]
                                out = torch.tensor(out).permute(2,0,1).unsqueeze(0).to(device) # torch.tensor([1, 3, h, w])
                                pred, pred_kappa = out[:, :3, :, :], out[:, 3:, :, :]
                                pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
                                
                                gt_key = {
                                    'albedo': 'albedo_values',
                                    'normal': 'normal_values',
                                    'specular': 'specular_values',
                                    'cross': 'cross_values' if pidx != 0 else 'static_cross_values',
                                    'parallel': 'parallel_values' if pidx != 0 else 'static_parallel_values'
                                }
                                
                                reference = img[None,...]
                                if 'cross' in task or 'parallel' in task:
                                    if pidx == 0:
                                        gt = (data_dict[gt_key[task]].to(device) + 1.0) / 2.0 # static_parallel_values is scaled to [-1,1] in dataloader, recover here
                                    else:
                                        gt = (data_dict[gt_key[task]][:,pidx-1].to(device) + 1.0) / 2.0 # parallel_values is scaled to [-1,1] in dataloader, recover here
                                elif 'forward_' in task:
                                    reference = img_irradiance[None,...].to(device) # debugging the lighting
                                    gt = img[None,...].to(device) # debugging the lighting
                                else:
                                    gt = (data_dict[gt_key[task]].to(device) + 1.0) / 2.0
                                gt_mask = data_dict['valid_mask_values'].to(device) if 'valid_mask_values' in data_dict.keys() else torch.ones_like(gt[:, :1, :, :], dtype=torch.bool)
                                pred_error = normal_utils.compute_cosine_error(pred, gt)
                                metrics = normal_utils.cross_verify_metrics(pred*0.5+0.5, gt*0.5+0.5, mask=gt_mask.repeat(1,3,1,1))
                                mosaic = vis_utils.visualize_albedo(results_dir, prefixs, reference, pred, pred_kappa, gt, gt_mask, pred_error)
                                
                                mosaics.append(mosaic)
                                metrics_dataset.append(metrics)
                                
                            
                        # concatnate mosaics
                        if len(mosaics) > 0:
                            mosaics = np.concatenate(mosaics, axis=0)  # (H*3, W, 3)
                            target_path = '%s/%s.png' % (os.path.join(path_save_out, task, 'mosaics'), f'{img_meta["obj"]}_cam{img_meta["cam"]}_l{img_meta["l"]}_all')
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            plt.imsave(target_path, mosaics)

                    # aggregate metrics across all processes
                    all_metrics = [None for _ in range(world_size)]
                    dist.all_gather_object(all_metrics, metrics_dataset)
                    if rank == 0:
                        metrics_dataset = sum(all_metrics, [])

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
                        metrics_out_path = os.path.join(os.path.dirname(path_save_out), f"{task}_metrics.txt")
                        with open(metrics_out_path, "w+") as f:
                            eval_text = f"Evaluation metrics:\n\
                            on dataset: {dataset_name}\n"
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
                    
    dist.barrier()
    dist.destroy_process_group()