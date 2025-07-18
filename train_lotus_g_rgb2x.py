#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from glob import glob
from easydict import EasyDict
import time

import accelerate
import datasets
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import transformers
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, gather, gather_object
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers

from pipeline import LotusGPipeline, LotusGMultistepsPipeline
from utils.image_utils import concatenate_images, colorize_depth_map
from utils.hypersim_dataset import get_hypersim_dataset_depth_normal
from utils.vkitti_dataset import VKITTIDataset, VKITTITransform, collate_fn_vkitti
from utils.lightstage_dataset import LightstageDataset, LightstageTransform, collate_fn_lightstage
from utils.empty_dataset import EmptyDataset

from eval import evaluation_depth, evaluation_normal, evaluation_material

import sys
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from rgbx.rgb2x.load_image import load_exr_image, load_ldr_image
# from ..rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
# from ..rgbx.rgb2x.load_image import load_exr_image, load_ldr_image

import tensorboard

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")
    
TOP5_STEPS_DEPTH = []
TOP5_STEPS_NORMAL = []

def load_image(img_path):
    # default rgb2x image loader

    if img_path.endswith(".exr"):
        photo = load_exr_image(img_path, tonemaping=True, clamp=True)
    elif (
        img_path.endswith(".png")
        or img_path.endswith(".jpg")
        or img_path.endswith(".jpeg")
    ):
        photo = load_ldr_image(img_path, from_srgb=False)
        
        # if resolution is over 1k, downsample to less than 1k
        if photo.shape[1] > 512: # photo in shape 3, H, W
            downsize = 512 / photo.shape[1]
            photo = torchvision.transforms.Resize((int(photo.shape[1] * downsize), int(photo.shape[2] * downsize)))(photo)

    return photo
        

def rgb2x(
    img_path,
    pipeline,
    accelerator,
    generator,
    inference_step = 50,
    num_samples = 1,
):
    # generator = torch.Generator(device="cuda").manual_seed(seed)
    
    photo = load_image(img_path).to(accelerator.device)

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    return_list = []
    for i in tqdm(range(num_samples), desc="Running Pipeline", leave=False):
        for aov_name in required_aovs:
            prompt = prompts[aov_name]
            generated_image = pipeline(
                prompt=prompt,
                photo=photo,
                num_inference_steps=inference_step,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            generated_image = torchvision.transforms.Resize(
                (old_height, old_width)
            )(generated_image)

            generated_image = (generated_image, f"Generated {aov_name} {i}")
            return_list.append(generated_image)

    return photo, return_list, prompts

def run_rgb2x_example_validation(
    pipeline,
    task,
    args,
    step,
    accelerator,
    generator,
    inference_step = 50,
    num_samples = 1,
    model_alias = 'model'
):
    
    validation_images = glob(os.path.join(args.validation_images, "*.jpg")) + glob(os.path.join(args.validation_images, "*.png"))
    validation_images = sorted(validation_images)
    
    pred_annos = []
    input_images = []

    # distributed inference
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)
    
    with distributed_state.split_between_processes(validation_images) as validation_batch:
        for i in tqdm(range(len(validation_batch)), desc=f"rgb2x quick_eval on {distributed_state.device}"):
            photo, preds, prompts = rgb2x(
                validation_batch[i], 
                pipeline, 
                accelerator,
                generator,
                inference_step, 
                num_samples
            )

            assert len(preds) == len(prompts), "Number of images and prompts should be equal."
            for j, (pred, prompt) in enumerate(zip(preds, prompts)):
                pred_annos.append(pred[0])
                        
                normal_outpath = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val', f'{model_alias}', f'{os.path.basename(validation_batch[i]).split(".")[0]}_{prompt}.jpg')
                os.makedirs(os.path.dirname(normal_outpath), exist_ok=True)
                pred[0].save(normal_outpath)
            
            # tensor to PIL and convert to RGB
            # photo = (photo * 0.5 + 0.5).clamp(0, 1)
            photo = (photo * 255).to(torch.uint8)
            photo = photo.permute(1, 2, 0).cpu().numpy()
            photo = Image.fromarray(photo)
            photo = photo.resize((pred[0].size), Image.LANCZOS)
            input_images.append(photo)

    # gather distributed results
    input_images = gather_object(input_images)
    pred_annos = gather_object(pred_annos)
        
    # based on tasks, split the pred_annos lists based on the number of tasks
    # e.g. pred_annos = [depth1, normal1, depth2, normal2, depth3, normal3] -> [[depth1, depth2, depth3], [normal1, normal2, normal3]]
    pred_annos = [pred_annos[i::len(prompts)] for i in range(len(prompts))]
    
    # Save output
    save_output = concatenate_images(input_images, *pred_annos if isinstance(pred_annos[0], list) else pred_annos)
    save_dir = os.path.join(args.output_dir,'eval', f'step{step:05d}', 'quick_val')
    os.makedirs(save_dir, exist_ok=True)
    save_output.save(os.path.join(save_dir, f'{model_alias}.jpg'))

def run_brdf_evaluation(pipeline, task, args, step, accelerator, generator, eval_first_n=10, model_alias=''):

    if step >= 0 and step % args.evaluation_steps == 0:
        test_data_dir = os.path.join(args.base_test_data_dir, task)
        dataset_split_path = "evaluation/dataset_brdf"
        eval_datasets = [('lightstage', 'test')]
        eval_dir = os.path.join(args.output_dir, 'eval', f'step{step:05d}')

        if 'rgb2x' in model_alias:
            gen_prediction = rgb2x
        elif 'lotus' in model_alias:
            gen_prediction = gen_lotus_normal
        elif 'dsine' in model_alias:
            gen_prediction = None

        eval_metrics = evaluation_material(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                                                gen_prediction=gen_prediction, pipeline=pipeline, accelerator=accelerator, generator=generator, 
                                                eval_datasets=eval_datasets,
                                                save_pred_vis=args.save_pred_vis, args=args, task=task, model_alias=model_alias)
        
    
def gen_lotus_normal(img_path, pipe, accelerator, prompt="", num_inference_steps=1):
    if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)

    with autocast_ctx:
        
        # Preprocess validation image
        # refer to run_lotus_example_validation, or train_lotus_g.py original implementation
        img = load_image(img_path).to(accelerator.device)[None,...]
        # img = img / 2.0 - 1.0 # [-1, 1]

        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

        pred_normal = pipe(
                        rgb_in=img, # [-1,1] 
                        task_emb=task_emb, 
                        prompt=prompt, 
                        num_inference_steps=num_inference_steps,
                        timesteps=[999], # fixed time step or use generator instead
                        output_type='pt',
                        ).images[0] # [0,1], (3,h,w)
        pred_normal = (pred_normal*2-1.0).unsqueeze(0) # [-1,1], (1,3,h,w)
    return pred_normal

def run_lotus_example_validation(pipeline, task, args, step, accelerator, generator, model_alias='model'):
    validation_images = glob(os.path.join(args.validation_images, "*.jpg")) + glob(os.path.join(args.validation_images, "*.png"))
    validation_images = sorted(validation_images)
    
    pred_annos = []
    input_images = []

    # distributed inference
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)
    
    with distributed_state.split_between_processes(validation_images) as validation_batch:
        for i in tqdm(range(len(validation_batch)), desc=f"lotus quick_eval on {distributed_state.device}"):
            if task == "depth":
                if torch.backends.mps.is_available():
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(accelerator.device.type)

                with autocast_ctx:
                    validation_image = Image.open(validation_batch[i]).convert("RGB")
                    input_images.append(validation_image)

                    # Preprocess validation image
                    validation_image = np.array(validation_image).astype(np.float32)
                    validation_image = torch.tensor(validation_image).permute(2,0,1).unsqueeze(0)
                    validation_image = validation_image / 127.5 - 1.0 
                    validation_image = validation_image.to(accelerator.device)

                    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                    # Run
                    pred_depth = pipeline(
                        rgb_in=validation_image, 
                        task_emb=task_emb,
                        prompt="", 
                        num_inference_steps=1, 
                        timesteps=[args.timestep],
                        generator=generator, 
                        output_type='np',
                        ).images[0]
                    
                    # Post-process the prediction
                    pred_depth = pred_depth.mean(axis=-1)
                    is_reverse_color = "disparity" in args.norm_type
                    depth_color = colorize_depth_map(pred_depth, reverse_color=is_reverse_color)
                    
                    pred_annos.append(depth_color)

            elif task == "normal":
                # for i in range(len(validation_images)):
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    with autocast_ctx:
                        validation_image = Image.open(validation_batch[i]).convert("RGB")
                        input_images.append(validation_image)

                        # Preprocess validation image
                        validation_image = np.array(validation_image).astype(np.float32)
                        validation_image = torch.tensor(validation_image).permute(2,0,1).unsqueeze(0)
                        validation_image = validation_image / 127.5 - 1.0 
                        validation_image = validation_image.to(accelerator.device)

                        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                        # Run
                        pred_normal = pipeline(
                            rgb_in=validation_image, 
                            task_emb=task_emb,
                            prompt="", 
                            num_inference_steps=1, 
                            timesteps=[args.timestep],
                            generator=generator,
                            ).images[0]
                        
                        normal_outpath = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val', f'{model_alias}', f'{os.path.basename(validation_batch[i]).split(".")[0]}.jpg')
                        os.makedirs(os.path.dirname(normal_outpath), exist_ok=True)
                        pred_normal.save(normal_outpath)
                        
                        pred_annos.append(pred_normal)
                        
            else:
                raise ValueError(f"Not Supported Task: {args.task_name}!")

    # gather distributed results
    input_images = gather_object(input_images)
    pred_annos = gather_object(pred_annos)

    # Save output
    save_output = concatenate_images(input_images, pred_annos)
    save_dir = os.path.join(args.output_dir,'eval', f'step{step:05d}', 'quick_val')
    os.makedirs(save_dir, exist_ok=True)
    save_output.save(os.path.join(save_dir, f'{model_alias}.jpg'))

def run_dsine_example_validation(normal_predictor, task, args, step, accelerator, generator, model_alias='dsine'):
    validation_images = glob(os.path.join(args.validation_images, "*.jpg")) + glob(os.path.join(args.validation_images, "*.png"))
    validation_images = sorted(validation_images)
    
    pred_annos = []
    input_images = []

    # distributed inference
    distributed_state = PartialState()
    normal_predictor.model.to(distributed_state.device)
    
    with distributed_state.split_between_processes(validation_images) as validation_batch:
        # print(f"[DEBUG] Rank {PartialState().process_index} out of {PartialState().num_processes}")
        # print(f"[RANK {distributed_state.process_index}] Received {len(validation_batch)} images out of {len(validation_images)} total")
        if task == "normal":
            for i in tqdm(range(len(validation_batch)), desc=f"dsine quick_eval on {distributed_state.device}"):
                if torch.backends.mps.is_available():
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(accelerator.device.type)

                with autocast_ctx:
                    # photo = cv2.imread(validation_images[i], cv2.IMREAD_COLOR)
                    # print(f"DEBUGGING Processing {validation_batch[i]}...")
                    validation_image = Image.open(validation_batch[i]).convert("RGB")
                    input_images.append(validation_image)
                    validation_image = np.array(validation_image).astype(np.float32)

                    # Use the model to infer the normal map from the input image
                    with torch.inference_mode():
                        normal = normal_predictor.infer_cv2(validation_image)[0] # Output shape: (3, H, W)
                        normal = (normal + 1.) / 2.  # Convert values to the range [0, 1]
                    normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0) # Output shape: (H, W, 3)
                    normal = Image.fromarray(normal)  # Convert to PIL Image
                    normal_outpath = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val', f'{model_alias}', f'{os.path.basename(validation_batch[i]).split(".")[0]}.jpg')
                    os.makedirs(os.path.dirname(normal_outpath), exist_ok=True)
                    normal.save(normal_outpath)

                    pred_annos.append(normal)
        else:
            raise ValueError(f"Not Supported Task: {args.task_name}!")

    # gather distributed results
    input_images = gather_object(input_images)
    pred_annos = gather_object(pred_annos)

    # Save output
    save_output = concatenate_images(input_images, pred_annos)
    save_dir = os.path.join(args.output_dir,'eval', f'step{step:05d}', 'quick_val')
    os.makedirs(save_dir, exist_ok=True)
    save_output.save(os.path.join(save_dir, f'{model_alias}.jpg'))

def run_evaluation(pipeline, task, args, step, accelerator):
    # Define prediction functions
    def gen_depth(rgb_in, pipe, prompt="", num_inference_steps=1):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            pred_depth = pipe(
                            rgb_in=rgb_input, 
                            task_emb=task_emb,
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            timesteps=[args.timestep],
                            output_type='np',
                            ).images[0]
            pred_depth = pred_depth.mean(axis=-1) # [0,1]
        return pred_depth

    def gen_normal(img, pipe, prompt="", num_inference_steps=1):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            pred_normal = pipe(
                            rgb_in=img, # [-1,1] 
                            task_emb=task_emb, 
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            timesteps=[args.timestep],
                            output_type='pt',
                            ).images[0] # [0,1], (3,h,w)
            pred_normal = (pred_normal*2-1.0).unsqueeze(0) # [-1,1], (1,3,h,w)
        return pred_normal

    if step > 0:
        if task == "depth":
            test_data_dir = os.path.join(args.base_test_data_dir, task)
            test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
            }
            if args.FULL_EVALUATION:
                print("==> Full Evaluation Mode!")
                test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
                "kitti": "configs/data_kitti_eigen_test.yaml",
                "scannet": "configs/data_scannet_val.yaml",
                "eth3d": "configs/data_eth3d.yaml",
                "diode": "configs/data_diode_all.yaml",
            }
            LEADER_DATASET = list(test_depth_dataset_configs.keys())[0]
            for dataset_name, config_path in test_depth_dataset_configs.items():
                eval_dir = os.path.join(args.output_dir, f'evaluation-{step:05d}', task, dataset_name)
                test_dataset_config = os.path.join(test_data_dir, config_path)
                alignment_type = "least_square_disparity" if "disparity" in args.norm_type else "least_square"
                metric_tracker = evaluation_depth(eval_dir, test_dataset_config, test_data_dir, eval_mode="generate_prediction",
                            gen_prediction=gen_depth, pipeline=pipeline, save_pred_vis=args.save_pred_vis, alignment=alignment_type)
                print(dataset_name,',', 'abs_relative_difference: ', metric_tracker.result()['abs_relative_difference'], 'delta1_acc: ', metric_tracker.result()['delta1_acc'], 'delta2_acc: ', metric_tracker.result()['delta2_acc'])
                
                if dataset_name == LEADER_DATASET:
                    TOP5_STEPS_DEPTH.append((metric_tracker.result()['abs_relative_difference'], f"step-{step}"))
                    TOP5_STEPS_DEPTH.sort(key=lambda x: x[0])
                    if len(TOP5_STEPS_DEPTH) > 5:
                        TOP5_STEPS_DEPTH.pop()

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_scalar(f"depth_{dataset_name}/rel", metric_tracker.result()['abs_relative_difference'], step)
                        tracker.writer.add_scalar(f"depth_{dataset_name}/delta1", metric_tracker.result()['delta1_acc'], step)
            
            top_five_cycles = [cycle_name for _, cycle_name in TOP5_STEPS_DEPTH]
            print("Top Five:", top_five_cycles)

        elif task == "normal":
            test_data_dir = os.path.join(args.base_test_data_dir, task)
            dataset_split_path = "eval/dataset_normal"
            eval_datasets = [('nyuv2', 'test')]
            if args.FULL_EVALUATION:
                eval_datasets = [('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel'), ('oasis','val')]
            eval_dir = os.path.join(args.output_dir, f'evaluation-{step:05d}', task)
            eval_metrics = evaluation_normal(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                                                gen_prediction=gen_normal, pipeline=pipeline, eval_datasets=eval_datasets,
                                                save_pred_vis=args.save_pred_vis)
            
            LEADER_DATASET = eval_datasets[0][0]
            mean_value = eval_metrics[LEADER_DATASET]['mean'] if eval_metrics[LEADER_DATASET]['mean'] == eval_metrics[LEADER_DATASET]['mean'] else float('inf')
            TOP5_STEPS_NORMAL.append((mean_value, f"step-{step}"))
            TOP5_STEPS_NORMAL.sort(key=lambda x: x[0])
            if len(TOP5_STEPS_NORMAL) > 5:
                TOP5_STEPS_NORMAL.pop()
            
            top_five_cycles = [cycle_name for _, cycle_name in TOP5_STEPS_NORMAL]
            print("Top Five:", top_five_cycles)

            for dataset_name, metrics in eval_metrics.items():
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_scalar(f"normal_{dataset_name}/mean", metrics['mean'], step)
                        tracker.writer.add_scalar(f"normal_{dataset_name}/11.25", metrics['a3'], step)

        else:
                raise ValueError(f"Not Supported Task: {task}!")

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, step, eval_first_n=10):
    logger.info("Running validation for task: %s... " % args.task_name[0])
    task = args.task_name[0]

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
    # if args.use_lora:
        # https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters
        # https://huggingface.co/docs/transformers/main/en/peft
        # https://huggingface.co/docs/transformers/v4.43.2/peft
        # https://huggingface.co/docs/peft/main/en/package_reference/lora
        # disable lora adapter to validate the training is conducted on unet only
        # https://huggingface.co/docs/diffusers/v0.28.1/api/loaders/peft
        # unet.disable_adapters()
        # pass

    def wrap_pipeline_lotus(pretrained_model_name_or_path, unet, model_alias='model', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False):

        if reload_pretrained_unet:
            # reloading unet to preserver the pretrained results
            unet_save = unet
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
                low_cpu_mem_usage=False, device_map=None,
            ).to(accelerator.device, weight_dtype)
        elif disable_lora_on_reference:
            # when not reloading, check if need to disable lora adapters
            if isinstance(unet, nn.parallel.DistributedDataParallel):
                unet.module.disable_adapters() # multi-gpu
            else:
                unet.disable_adapters() # single gpu
        else:
            # use the finetuned unet
            pass

        # Load pipeline
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        scheduler.register_to_config(prediction_type=args.prediction_type)
        pipeline = LotusGPipeline.from_pretrained(
            pretrained_model_name_or_path,
            scheduler=scheduler,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
    
        # Run example-validation
        run_lotus_example_validation(pipeline, task, args, step, accelerator, generator, model_alias=model_alias)
    
        # Run evaluation
        if enable_eval:
            run_brdf_evaluation(pipeline, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias)
        
        del pipeline

        if reload_pretrained_unet:
            unet = unet_save
        elif disable_lora_on_reference:
            if isinstance(unet, nn.parallel.DistributedDataParallel):
                unet.module.enable_adapters()
            else:
                unet.enable_adapters()
        else:
            pass

    def wrap_pipeline_rgb2x(pretrained_model_name_or_path, unet, model_alias='model', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False):

        if reload_pretrained_unet:
            # reloading unet to preserver the pretrained results
            unet_save = unet
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
                low_cpu_mem_usage=False, device_map=None,
            ).to(accelerator.device, weight_dtype)
        elif disable_lora_on_reference:
            # when not reloading, check if need to disable lora adapters
            if isinstance(unet, nn.parallel.DistributedDataParallel):
                unet.module.disable_adapters() # multi-gpu
            else:
                unet.disable_adapters() # single gpu
        else:
            # use the finetuned unet
            pass

        pipeline_rgb2x = StableDiffusionAOVMatEstPipeline.from_pretrained(
            pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline_rgb2x.scheduler = DDIMScheduler.from_config(
            pipeline_rgb2x.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        pipeline_rgb2x.to(accelerator.device)
        pipeline_rgb2x.set_progress_bar_config(disable=True)
        
        if args.enable_xformers_memory_efficient_attention:
            pipeline_rgb2x.enable_xformers_memory_efficient_attention()
            
        # photo_path = '/home/jyang/projects/ObjectReal/external/lotus/datasets/quick_validation/00.png'
        run_rgb2x_example_validation(pipeline_rgb2x, task, args, step, accelerator, generator, model_alias=model_alias)

        if enable_eval:
            # Note that, the results may different from the example_validation when evaluation loads exr images.
            run_brdf_evaluation(pipeline_rgb2x, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias)

        del pipeline_rgb2x

        if reload_pretrained_unet:
            unet = unet_save
        elif disable_lora_on_reference:
            if isinstance(unet, nn.parallel.DistributedDataParallel):
                unet.module.enable_adapters()
            else:
                unet.enable_adapters()
        else:
            pass

    def wrap_pipeline_dsine(pretrained_model_name_or_path, model_alias='model', enable_eval=False):
        # https://github.com/baegwangbin/DSINE

        normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

        run_dsine_example_validation(normal_predictor, task, args, step, accelerator, generator, model_alias=model_alias)

        if enable_eval:
            run_brdf_evaluation(normal_predictor, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias)

    enable_eval = True
    if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
        if args.use_lora:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, model_alias='lotus_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
            # wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, model_alias='lotus_finetune_lora_disable', reload_pretrained_unet=False, disable_lora_on_reference=True, enable_eval=enable_eval)
        else:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, model_alias='lotus_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)

    elif 'zheng95z/rgb-to-x' in args.pretrained_model_name_or_path:
        if args.use_lora:
            wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, model_alias='rgb2x_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=True)
            # wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, model_alias='rgb2x_finetune_lora_disable', reload_pretrained_unet=False, disable_lora_on_reference=True, enable_eval=enable_eval)
        else:
            wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, model_alias='rgb2x_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=True)

    # generate pretrained results
    if step == 0:
        wrap_pipeline_dsine("hugoycj/DSINE-hub", model_alias='dsine', enable_eval=enable_eval)
        wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval) # default model output
        wrap_pipeline_lotus('jingheya/lotus-normal-g-v1-1', unet, model_alias='lotus_original', reload_pretrained_unet=True, enable_eval=enable_eval) # default model output
    else:
        pass
        
    # if args.use_lora:
        # https://huggingface.co/docs/diffusers/v0.28.1/api/loaders/peft
        # when using lora, remember to enable the adapters after disabling, otherwise, trainning will failed
        # unet.enable_adapters()
        # pass
        
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir_hypersim",
        type=str,
        default=None,
        help=(
            "A folder containing the training data for hypersim"
        ),
    )
    parser.add_argument(
        "--train_data_dir_vkitti",
        type=str,
        default=None,
        help=(
            "A folder containing the training data for vkitti"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--base_test_data_dir",
        type=str,
        default="datasets/eval/"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=["depth","normal"],
        nargs="+"
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help=("A set of images evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution_hypersim",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--resolution_vkitti",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--prob_hypersim",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--prob_vkitti",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prob_lightstage",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lightstage_lighting_augmentation",
        type=str,
        default="random8",
        choices=["random8", "random16", "hdri", "none"],
    )
    parser.add_argument(
        "--lightstage_original_augmentation_ratio",
        type=str,
        default="1:1",
        choices=["1:1", "1:0", "0:1"],
    )
    parser.add_argument(
        "--mix_dataset",
        action="store_true",
        help='Whether to mix the training data from hypersim and vkitti'
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=['instnorm','truncnorm','perscene_norm','disparity','trunc_disparity'],
        default='trunc_disparity',
        help='The normalization type for the depth prediction'
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--align_cam_normal",
        action="store_true",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--truncnorm_min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=1
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--evaluation_top_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The prediction_type that shall be used for training. ",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--FULL_EVALUATION", action="store_true")
    parser.add_argument("--save_pred_vis", action="store_true")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_lotus_g",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for training. If True, LoRA will be used for all models.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir_hypersim is None and args.train_data_dir_vkitti is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    
    if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path:
        
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
            class_embed_type="projection", projection_class_embeddings_input_dim=4,
            low_cpu_mem_usage=False, device_map=None,
        )
        
        # Replace the first layer to accept 8 in_channels. 
        _weight = unet.conv_in.weight.clone()
        _bias = unet.conv_in.bias.clone()
        _weight = _weight.repeat(1, 2, 1, 1) 
        _weight *= 0.5
        # unet.config.in_channels *= 2
        config_dict = EasyDict(unet.config)
        config_dict.in_channels *= 2
        unet._internal_dict = config_dict

        # new conv_in channel
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in =nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = nn.Parameter(_weight)
        _new_conv_in.bias = nn.Parameter(_bias)
        unet.conv_in = _new_conv_in
    elif 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None,
        )
    elif 'rgb-to-x' in args.pretrained_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None,
        )

        pipeline_rgb2x = StableDiffusionAOVMatEstPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
        )

        noise_scheduler = DDIMScheduler.from_config(
            pipeline_rgb2x.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        
    # Apply LoRA to all attention processors
    if args.use_lora:
        # https://huggingface.co/docs/peft/v0.8.0/en/package_reference/lora
        # https://huggingface.co/docs/diffusers/en/training/lora
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
        lora_rank = 8
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config, adapter_name = 'lora_lotus')
        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", in_channels=8)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    if args.use_lora:
        # Only train LoRA parameters
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # -----------------Loading Datasets-----------------
    # automatically fallback to no mix_dataset when only one dataset has positive probability
    db_prob = {
        "hypersim": args.prob_hypersim,
        "vkitti": args.prob_vkitti,
        "lightstage": args.prob_lightstage,
    }
    probs_sum = sum(db_prob.values())
    if probs_sum <= 0:
        assert probs_sum <= 0, (
            "All dataset probabilities are set to 0. If you want to use only Hypersim dataset, please set `mix_dataset` to False."
        )
    else:
        # normalize the probabilities to sum up to 1
        db_prob = {k: v / probs_sum for k, v in db_prob.items()}

    # when use args.mix_dataset, check if at least two datasets have positive probability
    # if not, set args.mix_dataset to False
    if args.mix_dataset:
        if sum(1 for v in db_prob.values() if v > 0) < 2:
            logger.warning(
                "Only one dataset has positive probability. Setting `mix_dataset` to False."
            )
            args.mix_dataset = False

    # Get the datasets and dataloaders.
    # -------------------- Dataset1: Hypersim --------------------
    if db_prob["hypersim"] > 0:
        print("Loading hypersim dataset...")
        tik = time.time()
        train_hypersim_dataset, preprocess_train_hypersim, collate_fn_hypersim = get_hypersim_dataset_depth_normal(
            args.train_data_dir_hypersim, args.resolution_hypersim, args.random_flip, 
            norm_type=args.norm_type, truncnorm_min=args.truncnorm_min, align_cam_normal=args.align_cam_normal
            )
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                train_hypersim_dataset = train_hypersim_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset_hypersim = train_hypersim_dataset.with_transform(preprocess_train_hypersim)

        train_dataloader_hypersim = torch.utils.data.DataLoader(
            train_dataset_hypersim,
            shuffle=True,
            collate_fn=collate_fn_hypersim,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True
        )
        print("Loading hypersim dataset takes: ", time.time()-tik)
    else:
        train_dataset_hypersim = EmptyDataset()
        train_dataloader_hypersim = torch.utils.data.DataLoader(train_dataset_hypersim)
        print("Skipping hypersim dataset as its probability is set to 0.")
    # -------------------- Dataset2: VKITTI --------------------
    if db_prob["vkitti"] > 0:
        print("Loading vkitti dataset...")
        tik = time.time()
        transform_vkitti = VKITTITransform(random_flip=args.random_flip)
        train_dataset_vkitti = VKITTIDataset(args.train_data_dir_vkitti, transform_vkitti, args.norm_type, truncnorm_min=args.truncnorm_min)
        train_dataloader_vkitti = torch.utils.data.DataLoader(
            train_dataset_vkitti, 
            shuffle=True,
            collate_fn=collate_fn_vkitti,
            batch_size=args.train_batch_size, 
            num_workers=args.dataloader_num_workers,
            pin_memory=True
            )
        print("Loading vkitti dataset takes: ", time.time()-tik)
    else:
        train_dataset_vkitti = EmptyDataset()
        train_dataloader_vkitti = torch.utils.data.DataLoader(train_dataset_hypersim)
        print("Skipping vkitti dataset as its probability is set to 0.")
    
    # -------------------- Dataset4: Lightstage --------------------
    if db_prob["lightstage"] > 0:
        print("Loading lightstage dataset...")
        tik = time.time()
        train_dataset_lightstage = LightstageDataset(split='train', tasks=args.task_name, ori_aug_ratio=args.lightstage_original_augmentation_ratio, lighting_aug=args.lightstage_lighting_augmentation)
        train_dataloader_lightstage = torch.utils.data.DataLoader(
            train_dataset_lightstage,
            shuffle=True,
            collate_fn=collate_fn_lightstage,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True
        )
        print("Loading lightstage dataset takes: ", time.time()-tik)
    else:
        train_dataset_lightstage = EmptyDataset()
        train_dataloader_lightstage = torch.utils.data.DataLoader(train_dataset_lightstage)
        print("Skipping lightstage dataset as its probability is set to 0.")

    # dataset summary
    db_num = {
        "hypersim": len(train_dataset_hypersim),
        "vkitti": len(train_dataset_vkitti),
        "lightstage": len(train_dataset_lightstage),
    }
    db_samples = int(sum([db_num[k] * db_prob[k] for k in db_num.keys()])) # total number of samples in the mixed dataset
    
    # Lr_scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader_hypersim) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(db_samples / args.gradient_accumulation_steps)
    assert args.max_train_steps is not None or args.num_train_epochs is not None, "max_train_steps or num_train_epochs should be provided"
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_lightstage, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader_hypersim, train_dataloader_vkitti, train_dataloader_lightstage, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader_hypersim) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(db_samples / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("task_name")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch) * total_batch_size

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples Hypersim = {len(train_dataset_hypersim)}")
    # logger.info(f"  Num examples VKITTI = {len(train_dataset_vkitti)}")
    # logger.info(f"  Num examples Lightstage = {len(train_dataset_lightstage)}")
    logger.info(f"  Using mix datasets: {args.mix_dataset}")
    logger.info(f"  Dataset Num Examples = {db_num}")
    logger.info(f"  Dataset probabilities: {db_prob}")
    logger.info(f"  Dataset augmentation: {args.lightstage_lighting_augmentation}, {args.lightstage_original_augmentation_ratio}")
    # logger.info(f"  Dataset alternation probability of Hypersim = {args.prob_hypersim}")
    # logger.info(f"  Dataset alternation probability of VKITTI = {args.prob_vkitti}")
    # logger.info(f"  Dataset alternation probability of Lightstage = {args.prob_lightstage}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Unet timestep = {args.timestep}")
    logger.info(f"  Task name: {args.task_name}")
    logger.info(f"  Is Full Evaluation?: {args.FULL_EVALUATION}")
    logger.info(f"Output Workspace: {args.output_dir}")

    assert (db_samples // args.train_batch_size ) * args.num_train_epochs >= args.max_train_steps, (
        "The number of samples in the mixed dataset is too small for the given training batch size and number of epochs. "
        "Please adjust the training batch size or the number of epochs."
    )

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    
    # if accelerator.is_main_process and args.validation_images is not None:
    if args.validation_images is not None: # enable distributed validation
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
            eval_first_n=args.evaluation_top_k,
        )
        pass
            
    for epoch in range(first_epoch, args.num_train_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        iter_hypersim = iter(train_dataloader_hypersim)
        iter_vkitti = iter(train_dataloader_vkitti)
        iter_lightstage = iter(train_dataloader_lightstage)

        train_loss = 0.0
        log_ann_loss = 0.0
        log_rgb_loss = 0.0

        # for _ in range(len(train_dataloader_hypersim)):
        for _ in range(db_samples):
            if args.mix_dataset:
                if random.random() < args.prob_hypersim:
                    batch = next(iter_hypersim)
                else:
                    # Important note:
                    # In our training process, the Hypersim dataset is larger than the VKITTI dataset
                    # Therefore, when the smaller VKITTI dataset is exhausted, we need to restart iterating from the beginning
                    try:
                        batch = next(iter_vkitti)
                    except StopIteration:
                        iter_vkitti = iter(train_dataloader_vkitti)
                        batch = next(iter_vkitti)
            else:
                if args.prob_hypersim > 0:
                    try:
                        batch = next(iter_hypersim)
                    except StopIteration:
                        continue # next epoch
                elif args.prob_vkitti > 0:
                    try:
                        batch = next(iter_vkitti)
                    except StopIteration:
                        continue # next epoch
                elif args.prob_lightstage > 0:
                    try:
                        batch = next(iter_lightstage)
                    except StopIteration:
                        continue # next epoch
                else:
                    raise ValueError("No dataset has positive probability. Please check your dataset probabilities.")

            with accelerator.accumulate(unet):
                # Convert images to latent space
                rgb_latents = vae.encode(
                    torch.cat((batch["pixel_values"],batch["pixel_values"]), dim=0).to(weight_dtype)
                    ).latent_dist.sample() # [2B, 4, h, w]
                rgb_latents = rgb_latents * vae.config.scaling_factor
                # Convert target_annotations to latent space
                assert len(args.task_name) == 1
                if args.task_name[0] == "depth":
                    TAR_ANNO = "depth_values"
                elif args.task_name[0] == "normal":
                    TAR_ANNO = "normal_values"
                else:
                    raise ValueError(f"Do not support {args.task_name[0]} yet. ")
                
                if 'rgb-to-x' in args.pretrained_model_name_or_path:
                    # negate the x channel to adapt to the pretrained weights
                    # the adjusted normals aligns to the lotus normal space
                    batch[TAR_ANNO][:, 0, :, :] *= -1 # [B, 3, h, w]
                else:
                    pass
                
                target_latents = vae.encode(
                    torch.cat((batch[TAR_ANNO],batch["pixel_values"]), dim=0).to(weight_dtype)
                    ).latent_dist.sample() # [2B, 4, h, w]
                target_latents = target_latents * vae.config.scaling_factor

                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    bsz = target_latents.shape[0]
                    num_tasks = 1
                    bsz_per_task = int(bsz/(num_tasks+1))

                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    bsz = target_latents.shape[0]
                    num_tasks = 1
                    bsz_per_task = int(bsz/(num_tasks+1))
                    
                    rgb_latents = rgb_latents[:bsz_per_task] # [B, 4, h, w], use bsz_per_task here, since the batch sample can be smaller than args.train_batch_size
                    target_latents = target_latents[:bsz_per_task] # [B, 4, h, w]
                    bsz = rgb_latents.shape[0] # update bsz to the real batch size

                # Get the valid mask for the latent space
                valid_mask_for_latent = batch.get("valid_mask_values", None)
                if args.task_name[0] == "depth" and valid_mask_for_latent is not None:
                    sky_mask_for_latent = batch.get("sky_mask_values", None)
                    valid_mask_for_latent = valid_mask_for_latent + sky_mask_for_latent
                if valid_mask_for_latent is not None:
                    valid_mask_for_latent = valid_mask_for_latent.bool()
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down_anno = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool()
                    valid_mask_down_anno = valid_mask_down_anno.repeat((1, 4, 1, 1))
                else:
                    valid_mask_down_anno = torch.ones_like(target_latents[:bsz_per_task]).to(target_latents.device).bool()
                
                valid_mask_down_rgb = torch.ones_like(target_latents[bsz_per_task:]).to(target_latents.device).bool()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target_latents) # [2B, 4, h, w]
                
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (target_latents.shape[0], target_latents.shape[1], 1, 1), device=target_latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                
                # Set timestep
                # if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path:
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    # original single step
                    timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz)
                    timesteps = timesteps.long()
                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    # multiple timesteps
                    # TODO: https://github.com/prs-eth/Marigold/blob/b1dffaaa3f7a2fe406b3bb9dd3c358b30da66060/src/trainer/marigold_normals_trainer.py#L235
                    if args.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device, generator=generator).long() # [2B,]
                    # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device).long() # [2B,]
                    pass
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(target_latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps) # [2B, 4, h, w]

                # Concatenate rgb and depth
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    unet_input = torch.cat(
                        [rgb_latents, noisy_latents], dim=1
                    ) # [2B, 8, h, w]
                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    unet_input = torch.cat(
                        [noisy_latents, rgb_latents], dim=1
                    ) # [2B, 8, h, w]

                # Get the empty text embedding for conditioning
                # if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path:
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    prompt = ""
                    text_inputs = tokenizer(
                        prompt,
                        padding="do_not_pad",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    # prompt = "Albedo (diffuse basecolor)"
                    prompt = "Camera-space Normal" # checked, this is correct during the training
                    # prompt = ""
                    # from StableDiffusionAOVMatEstPipeline::_encode_prompt
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids.to(target_latents.device)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]
                encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

                # Predict
                # if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path:
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    # Get the task embedding
                    task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(accelerator.device)
                    task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz_per_task, 1)
                    task_emb_rgb = torch.tensor([0, 1]).float().unsqueeze(0).to(accelerator.device)
                    task_emb_rgb = torch.cat([torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1).repeat(bsz_per_task, 1)
                    task_emb = torch.cat((task_emb_anno, task_emb_rgb), dim=0)

                    # lotus used class_embed_type="projection" in the unet
                    model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False, class_labels=task_emb)[0] # [bsz, 8, h, w]
                
                    # Get the target for loss
                    # lotus use x0-prediction claimed in paper (https://arxiv.org/pdf/2409.18124)
                    target = target_latents # x0 prediction in lotus

                    # Compute loss
                    anno_loss = F.mse_loss(model_pred[:bsz_per_task][valid_mask_down_anno].float(), target[:bsz_per_task][valid_mask_down_anno].float(), reduction="mean")
                    rgb_loss = F.mse_loss(model_pred[bsz_per_task:][valid_mask_down_rgb].float(), target[bsz_per_task:][valid_mask_down_rgb].float(), reduction="mean")
                    loss = anno_loss + rgb_loss

                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    # ValueError: class_labels should be provided when num_class_embeds > 0
                    # rgb2x uses prompt to control instead of class_labels
                    # makesure unet_input uses the correct component,
                    # model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0] # [2B, 4, h, w]
                    model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0] # [B, 4, h, w]
                    
                    # rgb2x use v-prediction claimed in paper (https://arxiv.org/pdf/2405.00666) Sec. 4. 
                    # texnet/train_controlnet.py has v_prediction implementation
                    # reference: https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_normals_trainer.py
                    assert noise_scheduler.prediction_type == 'v_prediction', "Make sure the noise_scheduler is configured to use v_prediction."
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)

                    # Compute loss
                    anno_loss = F.mse_loss(model_pred[valid_mask_down_anno].float(), target[valid_mask_down_anno].float(), reduction="mean")
                    rgb_loss = 0*anno_loss # no rgb loss in rgb2x
                    loss = anno_loss + rgb_loss

                # Gather loss
                avg_anno_loss = accelerator.gather(anno_loss.repeat(args.train_batch_size)).mean()
                log_ann_loss += avg_anno_loss.item() / args.gradient_accumulation_steps
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                train_loss = log_ann_loss + log_rgb_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"SL": loss.detach().item(), 
                    "SL_A": anno_loss.detach().item(), 
                    "SL_R": rgb_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                "anno_loss": log_ann_loss,
                                "rgb_loss": log_rgb_loss},
                                 step=global_step)
                train_loss = 0.0
                log_ann_loss = 0.0
                log_rgb_loss = 0.0

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps
                
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                        if args.use_lora:
                            unwrapped_unet = unwrap_model(unet)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet, adapter_name='lora_lotus')
                            )
                            
                            LotusGPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )
                
                if global_step % validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        eval_first_n=args.evaluation_top_k,
                    )
                    pass

                    if accelerator.is_main_process:
                        # collect all and save the validation images
                        # TODO: here
                        pass

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)

        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)
        
        if args.use_lora:
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet, adapter_name='lora_lotus')
            )
            LotusGPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()