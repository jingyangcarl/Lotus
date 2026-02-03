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
from rgbx.x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
from rgbx.rgb2x.load_image import load_exr_image, load_ldr_image
# from ..rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
# from ..rgbx.rgb2x.load_image import load_exr_image, load_ldr_image
# from ..rgbx.x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline

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
        # if photo.shape[1] > 512: # photo in shape 3, H, W
        #     downsize = 512 / photo.shape[1]
        #     photo = torchvision.transforms.Resize((int(photo.shape[1] * downsize), int(photo.shape[2] * downsize)))(photo)

    return photo
        

def rgb2x(
    img_path,
    pipeline,
    accelerator = None,
    generator = None,
    inference_step = 50,
    num_samples = 1,
    img_rgb = None,
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance", "specular", "cross", "parallel"]
):
    if generator is None:
        generator = torch.Generator(device="cuda")
        
    if accelerator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = accelerator.device
    
    if img_rgb is None:
        photo = load_image(img_path).to(device)
    else:
        photo = img_rgb.to(device)

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height, old_width = photo.shape[1], photo.shape[2]
    new_height, new_width = old_height, old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height, new_width = max_side, int(max_side / radio)
    else:
        new_width, new_height = max_side, int(max_side * radio)
    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width, new_height = new_width // 8 * 8, new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
        
        "specular": "Specular Albedo",
        "cross": "Cross Polarization",
        "parallel": "Parallel Polarization"
    }
    
    # remove aovs not in required_aovs
    prompts = {k: v for k, v in prompts.items() if k in required_aovs}

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


def x2rgb(
    rgb_path, # only for visual
    albedo_path,
    normal_path,
    roughness_path,
    metallic_path,
    irradiance_path,
    prompt,
    pipeline,
    accelerator,
    generator,
    inference_step = 50,
    num_samples = 1,
    img_rgb = None,
    img_albedo = None,
    img_normal = None,
    img_roughness = None,
    img_metallic = None,
    img_irradiance = None,
):
    if generator is None:
        generator = torch.Generator(device="cuda")
        
    if accelerator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = accelerator.device
        
    if img_rgb is None:
        rgb_image = load_image(rgb_path).to(device)
    else:
        rgb_image = img_rgb.to(device)

    if img_albedo is None:
        albedo_image = load_image(albedo_path).to(device)
    else:
        albedo_image = img_albedo.to(device)

    if img_normal is None:
        normal_image = load_image(normal_path).to(device)
    else:
        normal_image = img_normal.to(device)

    if img_roughness is None:
        roughness_image = load_image(roughness_path).to(device)
    else:
        roughness_image = img_roughness.to(device)

    if img_metallic is None:
        metallic_image = load_image(metallic_path).to(device)
    else:
        metallic_image = img_metallic.to(device)

    if img_irradiance is None:
        irradiance_image = load_image(irradiance_path).to(device)
    else:
        irradiance_image = img_irradiance.to(device)

    # Set default height and width
    old_height, old_width = albedo_image.shape[1], albedo_image.shape[2]
    new_height, new_width = old_height, old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height, new_width = max_side, int(max_side / radio)
    else:
        new_width, new_height = max_side, int(max_side * radio)
    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width, new_height = new_width // 8 * 8, new_height // 8 * 8

    albedo_image = torchvision.transforms.Resize((new_height, new_width))(albedo_image)
    normal_image = torchvision.transforms.Resize((new_height, new_width))(normal_image)
    roughness_image = torchvision.transforms.Resize((new_height, new_width))(roughness_image)
    metallic_image = torchvision.transforms.Resize((new_height, new_width))(metallic_image)
    irradiance_image = torchvision.transforms.Resize((new_height, new_width))(irradiance_image)

    # Check if any of the input images are not None
    # and set the height and width accordingly
    gbuffer = [
        rgb_image,
        albedo_image,
        normal_image,
        roughness_image,
        metallic_image,
        irradiance_image,
    ]

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    return_list = []
    prompts = []
    # TODO: may need to update to support OLAT
    for i in range(num_samples):
        generated_image = pipeline(
            prompt='',
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=inference_step,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=required_aovs,
            # guidance_rescale=0.7,
            # output_type="np",
        ).images[0]
        
        generated_image = torchvision.transforms.Resize(
            (old_height, old_width)
        )(generated_image)

        generated_image = (generated_image, f"Generated Image {i}")
        return_list.append(generated_image)
        prompts.append(f"static") # TODO: olat_1, etc.

    return gbuffer, return_list, prompts

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
    validation_images = validation_images[:args.validation_top_k] if args.validation_top_k > 0 else validation_images
    
    # check if results is already generated
    # this function will be called multiple times during the inference, therefore skip if already done
    val_path = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val', f'{model_alias}')
    if os.path.isdir(val_path) and len(os.listdir(val_path)) >= len(validation_images) * num_samples:
        print(f"Skip rgb2x quick eval at step {step} for {model_alias}, already exists.")
        return
    
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
                        
                pred_outpath = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val', f'{model_alias}', f'{os.path.basename(validation_batch[i]).split(".")[0]}_{prompt}.jpg')
                os.makedirs(os.path.dirname(pred_outpath), exist_ok=True)
                pred[0].save(pred_outpath)
            
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

def run_x2rgb_example_validation(
    pipeline,
    task,
    args,
    step,
    accelerator,
    generator,
    inference_step = 50,
    num_samples = 1,
    model_alias = 'model',
    inverse_model_alias = 'inverse_model',
):
    out_path = os.path.join(args.output_dir, 'eval', f'step{step:05d}', 'quick_val')
    model_alias_ = model_alias if inverse_model_alias == '' else f'{model_alias}_via_{inverse_model_alias}'
    
    if not inverse_model_alias or inverse_model_alias == 'gt':
        print("Skip x2rgb quick eval for gt model due to missing input for in-the-wild images.")
        return
    if os.path.exists(os.path.join(out_path, f'{model_alias_}.jpg')):
        print(f"Skip x2rgb quick eval at step {step} for {model_alias_}, already exists.")
        return
    
    rgb2x_save_dir = os.path.join(args.output_dir, 'eval', f'step00000', 'quick_val', inverse_model_alias)
    if not os.path.exists(rgb2x_save_dir):
        print('Skip x2rgb quick eval due to missing rgb2x results at ', rgb2x_save_dir)
        print('Please run rgb2x evaluation first.')
        return
    print('loading from ', rgb2x_save_dir)
    
    validation_images = glob(os.path.join(args.validation_images, "*.jpg")) + glob(os.path.join(args.validation_images, "*.png"))
    validation_images = sorted(validation_images)
    validation_images = validation_images[:args.validation_top_k] if args.validation_top_k > 0 else validation_images
    
    pred_annos = []
    input_images = []

    # distributed inference
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    with distributed_state.split_between_processes(validation_images) as validation_batch:
        for i in tqdm(range(len(validation_batch)), desc=f"x2rgb quick_eval on {distributed_state.device}"):
            img_rgb = load_image(validation_batch[i]).to(accelerator.device)
            img_blk = torch.zeros_like(img_rgb) # fill empty channels with black matching the training setting
            img_wht = torch.ones_like(img_rgb)
            if task == "forward_gbuffer":
                gbuffer, preds, prompts = x2rgb(
                    validation_batch[i],
                    os.path.join(rgb2x_save_dir, f'{i:02d}_albedo.jpg'),
                    os.path.join(rgb2x_save_dir, f'{i:02d}_normal.jpg'),
                    os.path.join(rgb2x_save_dir, f'{i:02d}_specular.jpg'), # use specular here through the roughness channel. this should work as long as it matches training
                    '', '', # no metallic, irradiance
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_roughness.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_metallic.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_irradiance.jpg'),
                    'static',
                    pipeline,
                    accelerator,
                    generator,
                    inference_step, 
                    num_samples,
                    img_metallic=img_blk,
                    img_irradiance=img_wht,
                )
            elif task == "forward_gbuffer_albedo_only":
                gbuffer, preds, prompts = x2rgb(
                    validation_batch[i],
                    os.path.join(rgb2x_save_dir, f'{i:02d}_albedo.jpg'),
                    '', # no normal
                    '', # no roughness
                    '', '', # no metallic, irradiance
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_normal.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_roughness.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_metallic.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_irradiance.jpg'),
                    'static',
                    pipeline,
                    accelerator,
                    generator,
                    inference_step, 
                    num_samples,
                    img_normal=img_blk,
                    img_roughness=img_blk,
                    img_metallic=img_blk,
                    img_irradiance=img_wht,
                )
            elif task == "forward_polarization":
                gbuffer, preds, prompts = x2rgb(
                    validation_batch[i],
                    os.path.join(rgb2x_save_dir, f'{i:02d}_cross.jpg'),
                    os.path.join(rgb2x_save_dir, f'{i:02d}_parallel.jpg'),
                    '', '', '', # no roughness, metallic, irradiance
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_roughness.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_metallic.jpg'),
                    # os.path.join(rgb2x_save_dir, f'{i:02d}_irradiance.jpg'),
                    'static',
                    pipeline,
                    accelerator,
                    generator,
                    inference_step, 
                    num_samples,
                    img_roughness=img_blk,
                    img_metallic=img_blk,
                    img_irradiance=img_wht,
                )

            for j, (pred, prompt) in enumerate(zip(preds, prompts)):
                pred_annos.append(pred[0])

                pred_outpath = os.path.join(out_path, f'{model_alias_}', f'{os.path.basename(validation_batch[i]).split(".")[0]}_{prompt}.jpg')
                os.makedirs(os.path.dirname(pred_outpath), exist_ok=True)
                pred[0].save(pred_outpath)
            
            # tensor to PIL and convert to RGB
            photo = gbuffer[0] # rgb image
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
    save_output.save(os.path.join(save_dir, f'{model_alias_}.jpg'))

def run_brdf_evaluation(pipeline, task, args, step, accelerator, generator, eval_first_n=10, model_alias='', inverse_model_alias=''):

    if step > 0 and step % args.evaluation_steps == 0:
        test_data_dir = os.path.join(args.base_test_data_dir, task)
        dataset_split_path = "evaluation/dataset_brdf"
        eval_datasets = [('lightstage', 'test')]
        eval_dir = os.path.join(args.output_dir, 'eval', f'step{step:05d}')

        if 'rgb2x' in model_alias:
            gen_prediction = rgb2x
        elif 'x2rgb' in model_alias:
            gen_prediction = x2rgb
        elif 'lotus' in model_alias:
            gen_prediction = gen_lotus_normal
        elif 'dsine' in model_alias:
            gen_prediction = None

        eval_metrics = evaluation_material(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                                                gen_prediction=gen_prediction, pipeline=pipeline, accelerator=accelerator, generator=generator, 
                                                eval_datasets=eval_datasets,
                                                save_pred_vis=args.save_pred_vis, args=args, task=task, model_alias=model_alias, inverse_model_alias=inverse_model_alias)

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
    validation_images = validation_images[:args.validation_top_k] if args.validation_top_k > 0 else validation_images
    
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
    validation_images = validation_images[:args.validation_top_k] if args.validation_top_k > 0 else validation_images
    
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

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, step, eval_first_n=10, unet_fr=None):
    task = args.task_name[0]
    logger.info("Running validation for task: %s... " % task)

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

    def wrap_pipeline_lotus(pretrained_model_name_or_path, unet, task, model_alias='model', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False):

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

    def wrap_pipeline_rgb2x(pretrained_model_name_or_path, unet, task, model_alias='model', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False):

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

        if 'output/' not in pretrained_model_name_or_path:
            pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained(
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
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
            )
        else:
            # for rgb2albedo only model, no unet is provided
            # TODO: this function is not compatible with reload_pretrained_unet and disable_lora_on_reference
            pipeline = StableDiffusionAOVMatEstPipeline.from_pretrained('zheng95z/rgb-to-x')
            pipeline.load_lora_weights(pretrained_model_name_or_path)
        pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
            
        # Run example-validation
        run_rgb2x_example_validation(pipeline, task, args, step, accelerator, generator, model_alias=model_alias)

        if enable_eval:
            # Note that, the results may different from the example_validation when evaluation loads exr images.
            run_brdf_evaluation(pipeline, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias)
            pass

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
        
    def wrap_pipeline_x2rgb(pretrained_model_name_or_path, unet, task, model_alias='model', inverse_model_alias='', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False):

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

        pipeline = StableDiffusionAOVDropoutPipeline.from_pretrained(
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
        pipeline.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
            
        # Run example-validation
        run_x2rgb_example_validation(pipeline, task, args, step, accelerator, generator, model_alias=model_alias, inverse_model_alias=inverse_model_alias)

        if enable_eval:
            # Note that, the results may different from the example_validation when evaluation loads exr images.
            run_brdf_evaluation(pipeline, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias, inverse_model_alias=inverse_model_alias)
            pass

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

    def wrap_pipeline_dsine(pretrained_model_name_or_path, task, model_alias='model', enable_eval=False):
        # https://github.com/baegwangbin/DSINE

        normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

        run_dsine_example_validation(normal_predictor, task, args, step, accelerator, generator, model_alias=model_alias)

        if enable_eval:
            run_brdf_evaluation(normal_predictor, task, args, step, accelerator, generator, eval_first_n=eval_first_n, model_alias=model_alias)

    # enable_eval = True
    enable_eval = False # TODO, will OOM for some reason.
    
    # generate pretrained results
    if step == 0:
        task = args.task_name[0] # need to reset
        tasks = [task] if task != 'inverse' else ['albedo', 'normal', 'specular', 'cross', 'parallel']
        for task in tasks:
            # disable the rgb2x_original as it's the same a step0 results from lora enabled and it's really heavy
            if 'normal' == task:
                wrap_pipeline_dsine("hugoycj/DSINE-hub", task, model_alias='dsine', enable_eval=enable_eval)
                wrap_pipeline_lotus('jingheya/lotus-normal-g-v1-1', unet, task, model_alias='lotus_original', reload_pretrained_unet=True, enable_eval=enable_eval) # default model output
                # wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval) # default model output
                pass
            elif 'albedo' == task:
                # wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
                pass
            elif 'specular' == task:
                # wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
                pass
            elif 'cross' == task:
                # wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
                pass
            elif 'parallel' == task:
                # wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
                pass
            elif 'forward_' in task:
                print("Generating inverse rendering results for forward rendering evaluation...")
                # inverse rendering
                assert len(tasks) == 1, "When testing forward rendering, we generate results from inverse first"
                if 'forward_gbuffer' == task:
                    inverse_tasks = ['albedo', 'normal', 'specular']
                elif 'forward_gbuffer_albedo_only' == task:
                    inverse_tasks = ['albedo']
                elif 'forward_polarization' == task:
                    inverse_tasks = ['cross', 'parallel']
                else:
                    raise ValueError(f"Not Supported Forward Rendering Task: {task}!")
                for inv_task in inverse_tasks:
                    if args.pretrained_inverse_model_path is not None and os.path.exists(args.pretrained_inverse_model_path):
                        ckpts = [dirname for dirname in os.listdir(args.pretrained_inverse_model_path) if 'checkpoint-' in dirname]
                        ckpts.sort(key=lambda x: int(x.split('checkpoint-')[-1]), reverse=True)
                        if len(ckpts) > 0:
                            inverse_model_name = args.pretrained_inverse_model_path.split('/')[-1].strip() + '-' + ckpts[0]
                            wrap_pipeline_rgb2x(os.path.join(args.pretrained_inverse_model_path, ckpts[0]), unet, inv_task, model_alias=inverse_model_name, reload_pretrained_unet=True, enable_eval=enable_eval)
                    wrap_pipeline_rgb2x('zheng95z/rgb-to-x', unet, inv_task, model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
                
                # forward rendering
                print("Generating forward rendering results...")
                wrap_pipeline_x2rgb('zheng95z/x-to-rgb', unet, task, model_alias='x2rgb_original', inverse_model_alias='rgb2x_original', reload_pretrained_unet=True, enable_eval=enable_eval)
    else:
        pass
    
    if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
        assert task in ['normal', 'albedo', 'depth', 'inverse'], f"{args.pretrained_model_name_or_path} pipeline support normal and depth estimation task."
        tasks = [task] if task != 'inverse' else ['albedo', 'normal', 'depth']
        if args.use_lora:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, task, model_alias='lotus_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
        else:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, task, model_alias='lotus_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
            
    elif 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
        assert task in ['normal', 'depth', 'inverse'], f"{args.pretrained_model_name_or_path} pipeline support normal and depth estimation task."
        if args.use_lora:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, task, model_alias='lotus_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
        else:
            wrap_pipeline_lotus(args.pretrained_model_name_or_path, unet, task, model_alias='lotus_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
            
    elif 'zheng95z/rgb-to-x' in args.pretrained_model_name_or_path:
        assert task in ['albedo', 'normal', 'specular', 'cross', 'parallel', 'inverse'], f"{args.pretrained_model_name_or_path} pipeline support albedo and normal estimation task."
        tasks = [task] if task != 'inverse' else ['albedo', 'normal', 'specular', 'cross', 'parallel']
        for task in tasks:
            if args.use_lora:
                wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, task, model_alias='rgb2x_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=True)
                if unet_fr is not None: wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet_fr, model_alias='x2rgb_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False)
                # wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, model_alias='rgb2x_finetune_lora_disable', reload_pretrained_unet=False, disable_lora_on_reference=True, enable_eval=enable_eval) # this same as original
            else:
                wrap_pipeline_rgb2x(args.pretrained_model_name_or_path, unet, task, model_alias='rgb2x_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=True)
                if unet_fr is not None: wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet_fr, model_alias='x2rgb_finetune_nolora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=False)

    elif 'zheng95z/x-to-rgb' in args.pretrained_model_name_or_path:
        assert task in ['forward_gbuffer', 'forward_gbuffer_albedo_only', 'forward_polarization'], f"{args.pretrained_model_name_or_path} pipeline support forward rendering task."
        tasks = [task]
        for task in tasks:
            if args.use_lora:
                if args.pretrained_inverse_model_path is not None and os.path.exists(args.pretrained_inverse_model_path):
                    # wrap_pipeline_rgb2x('output/relighting/fixalbedo_fixradiance/train-rgb2x-lora-albedo-bsz32_FR_warmup40000_check_albedo', None, model_alias='rgb2x_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                    # wrap_pipeline_rgb2x('output/relighting/train-rgb2x-lora-albedo-bsz32-noFR', unet, model_alias='rgb2x_finetune_lora_enable', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                    ckpts = [dirname for dirname in os.listdir(args.pretrained_inverse_model_path) if 'checkpoint-' in dirname]
                    ckpts.sort(key=lambda x: int(x.split('checkpoint-')[-1]), reverse=True)
                    if len(ckpts) > 0:
                        inverse_model_name = args.pretrained_inverse_model_path.split('/')[-1].strip() + '-' + ckpts[0]
                        wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_lora_enable', inverse_model_alias=inverse_model_name, reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_lora_enable', inverse_model_alias='rgb2x_original', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_lora_enable', inverse_model_alias='gt', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
            else:
                # wrap_pipeline_rgb2x('output/relighting/train-rgb2x-lora-albedo-bsz32-noFR', unet, model_alias='rgb2x_finetune_no_lora', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                if args.pretrained_inverse_model_path is not None and os.path.exists(args.pretrained_inverse_model_path):
                    ckpts = [dirname for dirname in os.listdir(args.pretrained_inverse_model_path) if 'checkpoint-' in dirname]
                    ckpts.sort(key=lambda x: int(x.split('checkpoint-')[-1]), reverse=True)
                    if len(ckpts) > 0:
                        inverse_model_name = args.pretrained_inverse_model_path.split('/')[-1].strip() + '-' + ckpts[0]
                        wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_nolora', inverse_model_alias=inverse_model_name, reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_nolora', inverse_model_alias='rgb2x_original', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)
                wrap_pipeline_x2rgb(args.pretrained_model_name_or_path, unet, task, model_alias='x2rgb_finetune_nolora', inverse_model_alias='gt', reload_pretrained_unet=False, disable_lora_on_reference=False, enable_eval=enable_eval)

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
        "--pretrained_inverse_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models. Only used for generating results at step0",
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
    )
    parser.add_argument(
        "--lightstage_lighting_augmentation_pair_n",
        type=int,
        default=2,
        help="The number of lighting pairs to use for lightstage augmentation. If set to 1, no paired-training will be enabled.",
    )
    parser.add_argument(
        "--lightstage_augmentation_pair_loss_enable",
        action="store_true",
        help="Whether to enable the paired training loss for lightstage augmentation.",
    )
    parser.add_argument(
        "--lightstage_original_augmentation_ratio",
        type=str,
        default="1:1:1",
    )
    parser.add_argument(
        "--lightstage_img_ext",
        type=str,
        default="jpg",
        choices=["jpg", "exr"],
        help="The image file extension for lightstage dataset.",
    )
    parser.add_argument(
        "--lightstage_use_cache",
        action="store_true",
        help="Whether to use cached lightstage data for faster training. Note: the cache may not exist initially. need to work with rewrite_cache option.",
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
        "--evaluation_olat_steps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--validation_top_k",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--evaluation_top_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--evaluation_skip_step0",
        action="store_true",
        help="Whether to skip the evaluation at step0.",
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
    # parser.add_argument(
    #     "--forward_rendering_warmup_steps",
    #     type=int,
    #     default=0,
    #     help="Number of steps to warmup the rendering loss. When set larger than total training steps, the rendering loss will be disabled.",
    # )
    parser.add_argument('--train_unet_from_scratch', action='store_true', help="Whether to train unet from scratch.")

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
        
        # include x2rgb
        # if args.forward_rendering_warmup_steps < args.max_train_steps:
        #     unet_fr = UNet2DConditionModel.from_pretrained(
        #         args.pretrained_model_name_or_path.replace('rgb-to-x', 'x-to-rgb'), subfolder="unet", revision=args.revision, variant=args.variant,
        #         low_cpu_mem_usage=False, device_map=None,
        #     )
        #     pipeline_x2rgb = StableDiffusionAOVDropoutPipeline.from_pretrained(
        #         args.pretrained_model_name_or_path.replace('rgb-to-x', 'x-to-rgb'),
        #     )
        #     noise_scheduler_fr = DDIMScheduler.from_config(
        #         pipeline_x2rgb.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        #     )
    elif 'x-to-rgb' in args.pretrained_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None,
        )
        pipeline_x2rgb = StableDiffusionAOVDropoutPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
        )
        noise_scheduler = DDIMScheduler.from_config(
            pipeline_x2rgb.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
        
    elif 'diffusion_renderer' in args.pretrained_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None,
        )
        
        # forward renderer
        unet_fr = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path.replace('-inverse-', '-forward-'), subfolder="unet", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False, device_map=None,
        )
        
    if args.train_unet_from_scratch:
        # https://github.com/huggingface/diffusers/discussions/8458
        config = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        unet = UNet2DConditionModel.from_config(config=config.config)
        
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
        
        # also add LoRA to unet_fr if exists
        if 'unet_fr' in locals():
            unet_fr_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            unet_fr.requires_grad_(False)
            unet_fr.add_adapter(unet_fr_lora_config, adapter_name = 'lora_lotus_fr')
            lora_layers_fr = filter(lambda p: p.requires_grad, unet_fr.parameters())

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    if 'unet_fr' in locals():
        unet_fr.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()

            if 'unet_fr' in locals():
                unet_fr.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    # Determine the model name based on the index and available models
                    model_name = "unet" if i == 0 else f"unet_fr"
                    
                    if args.use_lora:
                        # For LoRA models, save the complete model state (base + LoRA weights merged)
                        # This way we can load it back directly
                        try:
                            model.save_pretrained(os.path.join(output_dir, model_name))
                            accelerator.print(f"Successfully saved {model_name} with LoRA to {os.path.join(output_dir, model_name)}")
                        except Exception as e:
                            accelerator.print(f"Warning: Failed to save {model_name} with LoRA: {e}")
                    else:
                        # For non-LoRA models, save normally  
                        try:
                            model.save_pretrained(os.path.join(output_dir, model_name))
                            accelerator.print(f"Successfully saved {model_name} to {os.path.join(output_dir, model_name)}")
                        except Exception as e:
                            accelerator.print(f"Warning: Failed to save {model_name}: {e}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                
                # Determine the model name based on the index and available models
                model_name = "unet" if i == 0 else f"unet_fr"

                # load diffusers style into model
                if args.use_lora:
                    # For LoRA models, we need to load the checkpoint directly into the current model
                    # since the model already has LoRA adapters added
                    checkpoint_path = os.path.join(input_dir, model_name)
                    
                    # First try to load as a regular checkpoint (contains both base + LoRA weights)
                    try:
                        # Load the config
                        from diffusers import UNet2DConditionModel
                        import json
                        
                        config_path = os.path.join(checkpoint_path, "config.json")
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model.register_to_config(**config)
                        
                        # Load the weights directly - this should include LoRA weights
                        from safetensors import safe_open
                        import torch
                        
                        # Try loading from safetensors first
                        weights_path = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
                        if not os.path.exists(weights_path):
                            weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                        
                        if os.path.exists(weights_path):
                            if weights_path.endswith('.safetensors'):
                                state_dict = {}
                                with safe_open(weights_path, framework="pt", device="cpu") as f:
                                    for key in f.keys():
                                        state_dict[key] = f.get_tensor(key)
                            else:
                                state_dict = torch.load(weights_path, map_location="cpu")
                            
                            # Load the state dict into the model (this includes LoRA weights)
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            
                            if missing_keys:
                                accelerator.print(f"Warning: Missing keys when loading {model_name}: {len(missing_keys)} keys")
                            if unexpected_keys:
                                accelerator.print(f"Warning: Unexpected keys when loading {model_name}: {len(unexpected_keys)} keys")
                            
                            accelerator.print(f"Successfully loaded {model_name} checkpoint with LoRA from {checkpoint_path}")
                        else:
                            accelerator.print(f"Warning: No weights file found for {model_name} at {checkpoint_path}")
                            
                    except Exception as e:
                        accelerator.print(f"Warning: Failed to load {model_name} checkpoint: {e}")
                        # Fallback to original loading approach
                        try:
                            original_in_channels = model.config.in_channels if hasattr(model.config, 'in_channels') else 4
                            load_model = UNet2DConditionModel.from_pretrained(
                                input_dir, 
                                subfolder=model_name, 
                                in_channels=original_in_channels,
                                low_cpu_mem_usage=False, 
                                device_map=None
                            )
                            model.register_to_config(**load_model.config)
                            model.load_state_dict(load_model.state_dict(), strict=False)
                            del load_model
                        except Exception as fallback_e:
                            accelerator.print(f"Error: Both checkpoint loading methods failed for {model_name}: {fallback_e}")
                            
                else:
                    from diffusers import UNet2DConditionModel
                    # For non-LoRA models, load normally
                    try:
                        original_in_channels = model.config.in_channels if hasattr(model.config, 'in_channels') else 4
                        
                        load_model = UNet2DConditionModel.from_pretrained(
                            input_dir, 
                            subfolder=model_name, 
                            in_channels=original_in_channels,
                            low_cpu_mem_usage=False, 
                            device_map=None
                        )
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                        accelerator.print(f"Successfully loaded {model_name} checkpoint (non-LoRA)")
                    except Exception as e:
                        accelerator.print(f"Error: Failed to load {model_name} checkpoint: {e}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if 'unet_fr' in locals():
            unet_fr.enable_gradient_checkpointing()

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
        if 'unet_fr' in locals():
            optimizer_fr = optimizer_cls(
                lora_layers_fr,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
            # merge two optimizers
            optimizer.add_param_group(optimizer_fr.param_groups[0])
            del optimizer_fr
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

        # The best thing to do is to increase the num_workers slowly and stop once there is no more improvement in your training speed.
        # https://lightning.ai/docs/pytorch/stable/advanced/speed.html
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
        train_dataset_lightstage = LightstageDataset(
            split='train', 
            tasks=args.task_name, 
            ori_aug_ratio=args.lightstage_original_augmentation_ratio, 
            lighting_aug=args.lightstage_lighting_augmentation, 
            lighting_aug_pair_n=args.lightstage_lighting_augmentation_pair_n,
            use_cache=args.lightstage_use_cache
        )
        ctx = torch.multiprocessing.get_context("spawn")
        train_dataloader_lightstage = torch.utils.data.DataLoader(
            train_dataset_lightstage,
            multiprocessing_context=ctx,
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
    
    if 'unet_fr' in locals():
        unet_fr = accelerator.prepare(unet_fr)

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
    
    assert len(args.task_name) == 1
    task_curr = args.task_name[0]
    # if task_curr == 'forward_gbuffer' or task_curr == 'forward_polarization':
    #     args.evaluation_skip_step0 = False # always do evaluation at step 0 for forward rendering tasks
    
    # if accelerator.is_main_process and args.validation_images is not None:
    if args.validation_images is not None and not args.evaluation_skip_step0: # enable distributed validation
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
            unet_fr=(unet_fr if 'unet_fr' in locals() else None)
        )
    else:
        print("Skipping validation at step 0. Usually this take a while")
            
    for epoch in range(first_epoch, args.num_train_epochs):
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        iter_hypersim = iter(train_dataloader_hypersim)
        iter_vkitti = iter(train_dataloader_vkitti)
        iter_lightstage = iter(train_dataloader_lightstage)

        train_loss = 0.0
        log_ann_loss = 0.0
        log_rgb_loss = 0.0
        log_rgb_pair_loss = 0.0
        log_relighting_loss = 0.0
        
        total_samples = 0
        augmented_samples = 0

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
                
            if 'augmented' in batch:
                total_samples += batch["pixel_values"].shape[0]
                # augmented_samples += (batch["augmented"] != 'static').sum().item()
                augmented_samples += np.sum(np.array(batch["augmented"]) != 'static').item()

            with accelerator.accumulate(unet, unet_fr if 'unet_fr' in locals() else None):
                # Convert images to latent space
                rgb_latents = vae.encode(
                    torch.cat((batch["pixel_values"],batch["pixel_values"]), dim=0).to(weight_dtype)
                    ).latent_dist.sample() # [2B, 4, h, w]
                rgb_latents = rgb_latents * vae.config.scaling_factor
                # Convert target_annotations to latent space
                if task_curr == "depth":
                    TAR_ANNO = "depth_values"
                elif task_curr == "normal":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['stable-diffusion-2-base', 'lotus-normal-g-v1-1', 'rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for normal estimation'
                    TAR_ANNO = "normal_values"
                elif task_curr == "albedo":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['stable-diffusion-2-base', 'rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for albedo estimation'
                    TAR_ANNO = "albedo_values"
                elif task_curr == "specular":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['stable-diffusion-2-base', 'rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for specular estimation'
                    TAR_ANNO = "specular_values"
                elif task_curr == 'cross':
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['stable-diffusion-2-base', 'rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for polarization estimation'
                    # TAR_ANNO = "static_cross_values"
                    TAR_ANNO = "pixel_cross_values"
                elif task_curr == 'parallel':
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['stable-diffusion-2-base', 'rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for polarization estimation'
                    # TAR_ANNO = "static_parallel_values"
                    TAR_ANNO = "pixel_parallel_values"
                elif task_curr == "inverse":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['rgb-to-x'], f'model {args.pretrained_model_name_or_path} not supported for inverse rendering'
                    TAR_ANNOs = ["albedo_values", "normal_values", "specular_values", "pixel_cross_values", "pixel_parallel_values"]
                    TAR_ANNO = random.choice(TAR_ANNOs)
                elif task_curr == "forward_gbuffer":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['x-to-rgb'], f'model {args.pretrained_model_name_or_path} not supported for forward rendering via gbuffer data'
                    TAR_ANNO = "pixel_values"
                elif task_curr == "forward_gbuffer_albedo_only":
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['x-to-rgb'], f'model {args.pretrained_model_name_or_path} not supported for forward rendering via gbuffer data'
                    TAR_ANNO = "pixel_values"
                elif task_curr == 'forward_polarization':
                    assert args.pretrained_model_name_or_path.split('/')[-1] in ['x-to-rgb'], f'model {args.pretrained_model_name_or_path} not supported for forward rendering via polarization'
                    TAR_ANNO = "pixel_values"
                else:
                    raise ValueError(f"Do not support {task_curr} yet. ")
                
                # this section made alignment very confusing
                # disable for the lightstage dataset
                # when using the hypersim dataset, if not work, need to check the dataset
                if 'rgb-to-x' in args.pretrained_model_name_or_path:
                    # negate the x channel to adapt to the pretrained weights
                    # the adjusted normals aligns to the lotus normal space
                    if TAR_ANNO == "normal_values":
                        batch[TAR_ANNO][:, 0, :, :] *= -1 # [B, 3, h, w]
                        pass
                    else:
                        pass
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
                    
                elif 'x-to-rgb' in args.pretrained_model_name_or_path:
                    bsz = target_latents.shape[0]
                    num_tasks = 1
                    bsz_per_task = int(bsz/(num_tasks+1))
                    
                    black_img = torch.zeros_like(batch["pixel_values"])
                    albedo_latents = vae.encode(batch["albedo_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    normal_latents = vae.encode(batch["normal_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    specular_latents = vae.encode(batch["specular_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    static_cross_latents = vae.encode(batch["static_cross_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    static_parallel_latents = vae.encode(batch["static_parallel_values"].to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    irradiance_latents = F.interpolate(batch["pixel_irradiance_values"], size=albedo_latents.shape[-2:], mode='bilinear', align_corners=False)
                    # black_latents = vae.encode(black_img.to(weight_dtype)).latent_dist.sample() # this cause rendering looks green
                    black_latents = torch.zeros_like(rgb_latents[:bsz_per_task])
                    black_latents_scale = black_latents * vae.config.scaling_factor
                    
                    if task_curr == "forward_gbuffer":
                        gbuffer_latents = torch.cat((albedo_latents, normal_latents, specular_latents, black_latents_scale, irradiance_latents[:,:3]), dim=1) # [B, 15, h, w]
                    elif task_curr == "forward_gbuffer_albedo_only":
                        gbuffer_latents = torch.cat((albedo_latents, black_latents_scale, black_latents_scale, black_latents_scale, irradiance_latents[:,:3]), dim=1) # [B, 15, h, w]
                    elif task_curr == "forward_polarization":
                        gbuffer_latents = torch.cat((static_cross_latents, static_parallel_latents, black_latents_scale, black_latents_scale, irradiance_latents[:,:3]), dim=1) # [B, 15, h, w]
                    target_latents = target_latents[:bsz_per_task] # [B, 4, h, w]
                    bsz = black_latents.shape[0] # update bsz to the real batch size
                else:
                    raise ValueError(f"Do not support {args.pretrained_model_name_or_path} yet. ")
                    

                # Get the valid mask for the latent space
                valid_mask_for_latent = batch.get("valid_mask_values", None)
                if task_curr == "depth" and valid_mask_for_latent is not None:
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
                    raise NotImplementedError("Noise offset is not implemented for lightstage_augmentation_pair_loss_enable. ")
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    raise NotImplementedError("Input perturbation is not implemented for lightstage_augmentation_pair_loss_enable. ")
                
                # Set timestep
                # if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path:
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    # original single step
                    timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz)
                    timesteps = timesteps.long()
                elif 'rgb-to-x' in args.pretrained_model_name_or_path or 'x-to-rgb' in args.pretrained_model_name_or_path:
                    # multiple timesteps
                    # TODO: https://github.com/prs-eth/Marigold/blob/b1dffaaa3f7a2fe406b3bb9dd3c358b30da66060/src/trainer/marigold_normals_trainer.py#L235
                    if args.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device, generator=generator).long() # [B,]
                    pass
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(target_latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # Concatenate rgb and depth
                if 'stable-diffusion-2-base' in args.pretrained_model_name_or_path or 'lotus-normal-g-v1-1' in args.pretrained_model_name_or_path:
                    unet_input = torch.cat(
                        [rgb_latents, noisy_latents], dim=1
                    ) # [2B, 8, h, w]
                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    unet_input = torch.cat(
                        [noisy_latents, rgb_latents], dim=1
                    ) # [B, 8, h, w]
                elif 'x-to-rgb' in args.pretrained_model_name_or_path:
                    unet_input = torch.cat(
                        [noisy_latents, gbuffer_latents], dim=1
                    ) # [B, 23, h, w]

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
                elif 'rgb-to-x' in args.pretrained_model_name_or_path or 'x-to-rgb' in args.pretrained_model_name_or_path:
                    if task_curr == "normal":
                        prompt = "Camera-space Normal" # checked, this is correct during the training
                    elif task_curr == "albedo":
                        prompt = "Albedo (diffuse basecolor)" # checked, this is correct during the training
                    elif task_curr == "specular":
                        prompt = "Specular Albedo" # not trained
                    elif task_curr == "cross":
                        prompt = "Cross Polarization" # not trained
                    elif task_curr == "parallel":
                        prompt = "Parallel Polarization" # not trained
                    elif task_curr == "inverse":
                        if TAR_ANNO == "normal_values":
                            prompt = "Camera-space Normal"
                        elif TAR_ANNO == "albedo_values":
                            prompt = "Albedo (diffuse basecolor)"
                        elif TAR_ANNO == "specular_values":
                            prompt = "Specular Albedo"
                        elif TAR_ANNO == "static_cross_values" or TAR_ANNO == "pixel_cross_values":
                            prompt = "Cross Polarization"
                        elif TAR_ANNO == "static_parallel_values" or TAR_ANNO == "pixel_parallel_values":
                            prompt = "Parallel Polarization"
                        else:
                            raise ValueError(f"Do not support {TAR_ANNO} yet. ")
                    elif task_curr == "forward_gbuffer" or task_curr == "forward_gbuffer_albedo_only":
                        prompt = ""
                    elif task_curr == "forward_polarization":
                        prompt = ""
                    else:
                        raise ValueError(f"Do not support {task_curr} yet. ")
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
                    rgb_pair_loss = 0*anno_loss
                    loss = anno_loss + rgb_loss + rgb_pair_loss

                elif 'rgb-to-x' in args.pretrained_model_name_or_path:
                    
                    # use pair loss for lightstage augmentation
                    if args.lightstage_augmentation_pair_loss_enable:
                        
                        # batch["parallel_values"] in shape [B, pair, 3, h, w], 0 used for agumentation, use 1 for pair
                        parallel_latents = vae.encode(
                            batch["parallel_values"][:,-1,...].to(weight_dtype) # 
                        ).latent_dist.sample() # [B, 4, h, w]
                        parallel_latents = parallel_latents * vae.config.scaling_factor
                        noise_pair = torch.randn_like(target_latents) # [B, 4, h, w]
                        noise_pair_latents = noise_scheduler.add_noise(target_latents, noise_pair, timesteps)
                        
                        # build unet input pair
                        unet_input_pair = torch.cat([noise_pair_latents, parallel_latents], dim=1)
                        assert unet_input_pair.shape == unet_input.shape, f"unet_input_pair shape {unet_input_pair.shape} does not match unet_input shape {unet_input.shape}"
                    
                        # B->2B
                        unet_input = torch.cat([unet_input, unet_input_pair], dim=0)
                        timesteps = torch.cat([timesteps, timesteps], dim=0)
                        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
                        valid_mask_down_anno = torch.cat([valid_mask_down_anno, valid_mask_down_anno], dim=0)
                        target_latents = torch.cat([target_latents, target_latents], dim=0)
                        noise = torch.cat([noise, noise_pair], dim=0)
                    
                    # ValueError: class_labels should be provided when num_class_embeds > 0
                    # rgb2x uses prompt to control instead of class_labels
                    # makesure unet_input uses the correct component,
                    model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0] # [B, 4, h, w]
                    
                    # rgb2x use v-prediction claimed in paper (https://arxiv.org/pdf/2405.00666) Sec. 4. 
                    # texnet/train_controlnet.py has v_prediction implementation
                    # reference: https://github.com/prs-eth/Marigold/blob/main/src/trainer/marigold_normals_trainer.py
                    assert noise_scheduler.prediction_type == 'v_prediction', "Make sure the noise_scheduler is configured to use v_prediction."
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                    
                    # ------- add x0-consistency between paired halves -------
                    # utilities to extract per-t scalars in broadcast shape
                    def _extract(a, t, x_shape):
                        # a: [num_steps] tensor or buffer on the same device
                        return a.gather(0, t).reshape(-1, *([1] * (len(x_shape) - 1)))

                    # alphas_cumprod is used by DDPM-like schedulers
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(unet_input.device)
                    alpha_t = _extract(torch.sqrt(alphas_cumprod), timesteps, unet_input.shape)
                    sigma_t = _extract(torch.sqrt(1.0 - alphas_cumprod), timesteps, unet_input.shape)
                    denom = alpha_t**2 + sigma_t**2
                    rgb_x0_pred = (alpha_t * unet_input[:,:bsz_per_task] - sigma_t * model_pred) / denom  # [B,4,h,w]

                    # Compute loss
                    anno_loss = F.mse_loss(model_pred[valid_mask_down_anno].float(), target[valid_mask_down_anno].float(), reduction="mean")
                    rgb_loss = torch.tensor(0.0).to(anno_loss.device) # no rgb loss in rgb2x
                    rgb_pair_loss = F.mse_loss(rgb_x0_pred[:bsz_per_task][valid_mask_down_anno[:bsz_per_task]].float(), rgb_x0_pred[bsz_per_task:][valid_mask_down_anno[bsz_per_task:]].float(), reduction="mean") if args.lightstage_augmentation_pair_loss_enable else 0*anno_loss
                    rendering_loss = torch.tensor(0.0).to(anno_loss.device)
                    loss = anno_loss + rgb_loss + rgb_pair_loss*0
                    
                    # forward renderer loss
                    # if 'unet_fr' in locals() and global_step+1 >= args.forward_rendering_warmup_steps:
                        
                    #     noise_fr = torch.randn_like(rgb_latents) # [B, 4, h, w]
                    #     timesteps_fr = torch.randint(0, noise_scheduler_fr.config.num_train_timesteps, (bsz,), device=target_latents.device, generator=generator).long() # [B,]
                    #     noisy_latents_fr = noise_scheduler_fr.add_noise(target_latents[:bsz], noise_fr, timesteps_fr)

                    #     # pass through the first stage unet_fr
                    #     with torch.no_grad():
                    #         prompts = {
                    #             "albedo": "Albedo (diffuse basecolor)",
                    #             "normal": "Camera-space Normal",
                    #             "roughness": "Roughness",
                    #             "metallic": "Metallicness",
                    #             "irradiance": "Irradiance (diffuse lighting)",
                    #         }
                    #         task_latents = []
                    #         for task in prompts.keys():
                                
                    #             if prompts[task] == prompt:
                    #                 # save computation
                    #                 task_latents.append(model_pred[:bsz].detach())
                    #                 continue
                                
                    #             text_inputs_ = tokenizer(
                    #                 task,
                    #                 padding="max_length",
                    #                 max_length=tokenizer.model_max_length,
                    #                 truncation=True,
                    #                 return_tensors="pt",
                    #             )
                    #             text_input_ids_ = text_inputs_.input_ids.to(target_latents.device)
                    #             encoder_hidden_states_ = text_encoder(text_input_ids_, return_dict=False)[0]
                    #             encoder_hidden_states_ = encoder_hidden_states_.repeat(bsz, 1, 1)
                            
                    #             # TODO: this cannot be used for trainning the forward right? as the output not the final prediction of different kinds
                    #             # TODO: maybe we can use x0 status
                    #             model_pred_ = unet(unet_input[:bsz], timesteps[:bsz], encoder_hidden_states_, return_dict=False)[0]
                                
                    #             if task == 'irradiance':
                    #                 # decode to rgb and resize to other branch resolution
                    #                 latent_hw = model_pred_.shape[-2:]
                    #                 model_pred_ = vae.decode((model_pred_.float() / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
                    #                 model_pred_ = F.interpolate(model_pred_, size=latent_hw, mode='bilinear', align_corners=False)
                    #                 model_pred_ = model_pred_.float() # should stays in [-1,1]
                                
                    #             task_latents.append(model_pred_)
                            
                    #         unet_fr_input = torch.cat([noisy_latents_fr] + task_latents, dim=1) # [B, 4*6, h, w]
                            
                    #     # forward rendering
                    #     text_inputs_ = tokenizer(
                    #         '',
                    #         padding="max_length",
                    #         max_length=tokenizer.model_max_length,
                    #         truncation=True,
                    #         return_tensors="pt",
                    #     )
                    #     text_input_ids_ = text_inputs_.input_ids.to(target_latents.device)
                    #     encoder_hidden_states_ = text_encoder(text_input_ids_, return_dict=False)[0]
                    #     encoder_hidden_states_ = encoder_hidden_states_.repeat(bsz, 1, 1)
                    #     model_fr_pred = unet_fr(unet_fr_input, timesteps_fr, encoder_hidden_states_, return_dict=False)[0] # [B, 4, h, w]

                    #     assert noise_scheduler_fr.config.prediction_type == 'v_prediction', "Make sure the noise_scheduler is configured to use v_prediction."
                    #     # target_fr = noise_scheduler_fr.get_velocity(rgb_latents, noise_fr, timesteps_fr)
                    #     target_fr = noise_scheduler_fr.get_velocity(target_latents[:bsz], noise_fr, timesteps_fr) #

                    #     # rendering_loss = F.mse_loss(model_pred_fr[valid_mask_down_anno[:bsz]].float(), rgb_latents[valid_mask_down_anno[:bsz]].float(), reduction="mean")
                    #     rendering_loss = F.mse_loss(model_fr_pred[valid_mask_down_anno[:bsz]].float(), target_fr[valid_mask_down_anno[:bsz]].float(), reduction="mean")
                    #     loss = loss + rendering_loss
                        
                elif 'x-to-rgb' in args.pretrained_model_name_or_path:
                    model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False)[0] # [B, 4, h, w]
                    
                    assert noise_scheduler.prediction_type == 'v_prediction', "Make sure the noise_scheduler is configured to use v_prediction."
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                    
                    anno_loss = F.mse_loss(model_pred[valid_mask_down_anno].float(), target[valid_mask_down_anno].float(), reduction="mean")
                    rgb_loss = torch.tensor(0.0).to(anno_loss.device) # no rgb loss in x2rgb
                    rgb_pair_loss = torch.tensor(0.0).to(target.device)
                    rendering_loss = torch.tensor(0.0).to(target.device)
                    loss = anno_loss + rgb_loss + rgb_pair_loss + rendering_loss

                # Gather loss
                avg_anno_loss = accelerator.gather(anno_loss.repeat(args.train_batch_size)).mean()
                log_ann_loss += avg_anno_loss.item() / args.gradient_accumulation_steps
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                avg_rgb_pair_loss = accelerator.gather(rgb_pair_loss.repeat(args.train_batch_size)).mean() if args.lightstage_augmentation_pair_loss_enable else torch.tensor(0.0)
                log_rgb_pair_loss = avg_rgb_pair_loss.item() / args.gradient_accumulation_steps if args.lightstage_augmentation_pair_loss_enable else 0.0
                avg_relighting_loss = accelerator.gather(rendering_loss.repeat(args.train_batch_size)).mean()
                log_relighting_loss = avg_relighting_loss.item() / args.gradient_accumulation_steps
                train_loss = log_ann_loss + log_rgb_loss + log_rgb_pair_loss + log_relighting_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"L": loss.detach().item(), 
                    "L_A": anno_loss.detach().item(), 
                    "L_R": rgb_loss.detach().item(), 
                    "L_RP": rgb_pair_loss.detach().item(),
                    "L_FR": rendering_loss.detach().item(),
                    "R_AG": augmented_samples / total_samples if total_samples > 0 else 0.0,
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                "anno_loss": log_ann_loss,
                                "rgb_loss": log_rgb_loss,
                                "rgb_pair_loss": log_rgb_pair_loss,
                                "relighting_loss": log_relighting_loss,
                                'lr': lr_scheduler.get_last_lr()[0],},
                                 step=global_step)
                train_loss = 0.0
                log_ann_loss = 0.0
                log_rgb_loss = 0.0
                log_rgb_pair_loss = 0.0
                log_relighting_loss = 0.0

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
                        accelerator.save_state(save_path) # TODO: this will overwrite unet when unet_fr is enabled
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
                            
                            # if 'unet_fr' in locals():
                            #     unwrapped_unet_fr = unwrap_model(unet_fr)
                            #     unet_fr_lora_state_dict = convert_state_dict_to_diffusers(
                            #         get_peft_model_state_dict(unwrapped_unet_fr, adapter_name='lora_lotus_fr')
                            #     )
                                
                            #     LotusGPipeline.save_lora_weights(
                            #         save_directory=save_path,
                            #         unet_lora_layers=unet_fr_lora_state_dict,
                            #         safe_serialization=True,
                            #     )
                
                if global_step % validation_steps == 0:
                    
                    # save out augmented images
                    if 'parallel_values' in batch:
                        
                        visual_mosaic_outpath = os.path.join(args.output_dir, 'visual', 'aug_parallel', 'mosaic', f'step{global_step:05d}.png')
                        os.makedirs(os.path.dirname(visual_mosaic_outpath), exist_ok=True)

                        mosaics = []
                        mosaics.append(batch['pixel_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['pixel_irradiance_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['albedo_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['normal_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['normal_c2w_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['normal_ls_w2c_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['normal_ls_c2w_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['specular_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['static_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['static_cross_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(batch['static_parallel_values'].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics.append(torch.ones_like(batch['static_values']).cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3], irradiance for static is not available, fill with 1
                        for j in range(batch['parallel_values'].shape[1]):
                            mosaics.append(batch['parallel_values'][:,j].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                            mosaics.append(batch['irradiance_values'][:,j].cpu().numpy().transpose(0, 2, 3, 1)) # [B, h, w, 3]
                        mosaics = np.concatenate(mosaics, axis=2) # [B, h, w*7, 3]
                        mosaic = np.concatenate(mosaics, axis=0) # [B*h, w*7, 3]
                        mosaic = (mosaic + 1.0) * 0.5 # scale from [-1, 1] to [0, 1]
                        mosaic = Image.fromarray((mosaic * 255).astype(np.uint8))
                        mosaic.save(visual_mosaic_outpath)

                        for b in range(len(batch['parallel_values'])):
                            objname = '_'.join(batch['static_paths'][b].split('/')[-3:-1]).split('.')[0]
                            visual_sep_outpath = os.path.join(args.output_dir, 'visual', 'aug_parallel', 'separate', f'step{global_step:05d}', f'{b:02d}_{objname}')
                            os.makedirs(visual_sep_outpath, exist_ok=True)
                            
                            for j in range(batch['parallel_values'][b].shape[0]):
                                img = batch['parallel_values'][b][j].cpu().numpy().transpose(1, 2, 0)
                                img = (img + 1.0) * 0.5 # scale from [-1, 1] to [0, 1]
                                img = Image.fromarray((img * 255).astype(np.uint8))
                                img.save(os.path.join(visual_sep_outpath, f"augmented_olat_parallel_pair{j:02d}.png"))
                                
                                img = batch['irradiance_values'][b][j].cpu().numpy().transpose(1, 2, 0)
                                img = (img + 1.0) * 0.5 # scale from [-1, 1] to [0, 1]
                                img = Image.fromarray((img * 255).astype(np.uint8))
                                img.save(os.path.join(visual_sep_outpath, f"augmented_olat_irradiance_pair{j:02d}.png"))
                                
                            
                            # img = batch['parallel_values_hstacked'][b].cpu().numpy().transpose(1, 2, 0) # no need to scale this, as this is not scaled in the dataloader
                            # img = Image.fromarray((img * 255).astype(np.uint8))
                            # img.save(os.path.join(visual_sep_outpath, f"augmented_olat_parallel_hstacked.png"))
                            
                            
                            # the following is only used for rgb2x and x2rgb is also enabled
                            # if 'unet_fr' in locals() and global_step >= args.forward_rendering_warmup_steps:
                                
                            #     # visual of forward rendering prediction
                            #     fr_pred = model_fr_pred[b:b+1]
                            #     alpha_t_ = _extract(torch.sqrt(alphas_cumprod), timesteps[b:b+1], fr_pred.shape)
                            #     sigma_t_ = _extract(torch.sqrt(1.0 - alphas_cumprod), timesteps[b:b+1], fr_pred.shape)
                            #     denom_ = alpha_t_**2 + sigma_t_**2
                            #     fr_pred_x0 = (alpha_t_ * unet_input[b:b+1, :bsz_per_task] - sigma_t_ * fr_pred) / denom_  # [1,4,h,w]
                                
                            #     pred_decode = vae.decode((fr_pred_x0.float() / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
                            #     img = pred_decode[0].float().detach().cpu().numpy().transpose(1, 2, 0)
                            #     img = (img + 1.0) * 0.5 # scale from [-1, 1] to [0, 1]
                            #     img = Image.fromarray((img * 255).astype(np.uint8))
                            #     img.save(os.path.join(visual_outpath, f"pred_fr_static.png"))
                                
                            #     # visual of albedo prediction
                            #     albedo_pred = unet_fr_input[b:b+1, 4:8].float() # from rgb2x, therefore use default vae scaling_factor
                            #     alpha_t_ = _extract(torch.sqrt(alphas_cumprod), timesteps[b:b+1], albedo_pred.shape)
                            #     sigma_t_ = _extract(torch.sqrt(1.0 - alphas_cumprod), timesteps[b:b+1], albedo_pred.shape)
                            #     denom_ = alpha_t_**2 + sigma_t_**2
                            #     albedo_pred_x0 = (alpha_t_ * unet_input[b:b+1, :bsz_per_task] - sigma_t_ * albedo_pred) / denom_  # [1,4,h,w]
                                
                            #     # get the decoded albedo image
                            #     pred_decode = vae.decode((albedo_pred_x0.float() / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
                            #     img = pred_decode[0].float().detach().cpu().numpy().transpose(1, 2, 0)
                            #     img = (img + 1.0) * 0.5 # scale from [-1, 1] to [0, 1]
                            #     img = Image.fromarray((img * 255).astype(np.uint8))
                            #     img.save(os.path.join(visual_outpath, f"pred_ir_albedo.png"))
                            
                            # TODO: save out the olat hdri
                            
                    torch.cuda.empty_cache()
                    
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
                        unet_fr=(unet_fr if 'unet_fr' in locals() else None)
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
        pipeline.save_pretrained(os.path.join(args.output_dir, 'pipe'))

        if 'unet_fr' in locals():
            unet_fr = unwrap_model(unet_fr)
            pipeline_fr = LotusGPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet_fr,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline_fr.save_pretrained(os.path.join(args.output_dir, 'pipe_fr'))

        if args.use_lora:
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unet, adapter_name='lora_lotus')
            )
            LotusGPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            
            if 'unet_fr' in locals():
                unet_fr_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unet_fr, adapter_name='lora_lotus_fr')
                )
                LotusGPipeline.save_lora_weights(
                    save_directory=os.path.join(args.output_dir, 'pipe_fr'),
                    unet_lora_layers=unet_fr_lora_state_dict,
                    safe_serialization=True,
                )

    accelerator.end_training()


if __name__ == "__main__":
    main()