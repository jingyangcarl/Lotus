import os
import pandas as pd
from torch.utils.data import Dataset
import imageio
import numpy as np
import socket
from scipy.spatial.transform import Rotation as R
import torch
import json
from tqdm.rich import tqdm
from torchvision import transforms
import cv2

class LightstageTransform:
    pass

class LightstageDataset(Dataset):

    def __init__(self, split='train', tasks='', ori_aug_ratio='1:1', lighting_aug='random8', lighting_aug_pair_n = 2, eval_first_n=None):

        assert split in ['train', 'test'], f'Invalid split: {split}'
        
        v = 'v1.3'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_half.json'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_debug.json'
        
        self.root_dir = '/labworking/Users_A-L/jyang/data/LightStageObjectDB'
        self.root_dir = '/home/jyang/data/LightStageObjectDB' # local cache, no IO bottle neck
        img_ext = 'jpg' # 'exr' or 'jpg' # TODO: jpg need to updated to compatible with negative values, running now
        # when use exr, the evaluation results may look different to the validation due to 
        # meta_data_path = f'{self.root_dir}/datasets/exr/train.json'
        # meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.json'
        # meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.csv'
        meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_fitting_512_ck.csv'
        # The code `self.dataset_dir` is accessing the `dataset_dir` attribute of the current object
        # (instance) in Python. This code is typically found within a class definition where `self`
        # refers to the current instance of the class. By accessing `self.dataset_dir`, the code is
        # retrieving the value stored in the `dataset_dir` attribute of the current object.
        self.dataset_dir = f'{self.root_dir}/datasets/{img_ext}/{v}/{v}_2'
        self.cam_dir = f'{self.root_dir}/Redline/exr/{v}/{v}_2/cameras'
        
        self.original_augmentation_ratio = ori_aug_ratio
        self.lighting_augmentation = lighting_aug
        self.lighting_augmentation_pair_n = lighting_aug_pair_n # number of pairs to generate for lighting augmentation, aiming on the same prediction
        
        # load json file
        metadata = []
        with open(meta_data_path) as f:
            if '.json' in meta_data_path:
                metadata = json.load(f)
            elif '.csv' in meta_data_path:
                metadata = pd.read_csv(f).to_dict(orient='records')
                
        # add a manual expansion here since the cropping's under processing
        if 'fitting' in meta_data_path:
            metadata_ = []
            expansion_counter = 0
            for row in tqdm(metadata, 'expanding metadata'):
                cross_dir_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross')
                paral_dir_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel')
                
                cross_img_path = sorted(os.listdir(cross_dir_path)) if os.path.isdir(cross_dir_path) else []
                paral_img_path = sorted(os.listdir(paral_dir_path)) if os.path.isdir(paral_dir_path) else []
                
                n_cross = len(cross_img_path)
                n_paral = len(paral_img_path)
                
                if n_cross == 350 and n_paral == 350:
                    # for l in range(350):
                    for l in range(10): # debug
                        row['l'] = l
                        
                        if self.original_augmentation_ratio == '1:1':
                            row_ = row.copy()
                            row_['aug'] = True
                            metadata_.append(row_)
                            row_ = row.copy()
                            row_['aug'] = False
                            metadata_.append(row_)
                        elif self.original_augmentation_ratio == '1:0':
                            row_ = row.copy()
                            row_['aug'] = False
                            metadata_.append(row_)
                        elif self.original_augmentation_ratio == '0:1':
                            row_ = row.copy()
                            row_['aug'] = True
                            metadata_.append(row_)
                        else:
                            raise NotImplementedError(f'Original augmentation ratio {ori_aug_ratio} is not supported')
                        
                    expansion_counter += 1
                    
                    # add a code to add all the olat images together here to make the -random8 faster, this only need to be done once
                    cross_olat_sum_path = cross_dir_path.replace('cross', 'cross_hdri')
                    paral_olat_sum_path = paral_dir_path.replace('parallel', 'parallel_hdri')
                    os.makedirs(cross_olat_sum_path, exist_ok=True)
                    os.makedirs(paral_olat_sum_path, exist_ok=True)
                    
                    hdri_list = [
                        'allwhite'
                    ]
                    
                    for hdri in hdri_list:
                        force_update = False
                        if not os.path.isfile(os.path.join(cross_olat_sum_path, f'{hdri}.exr')) or force_update: # if the hdri file already exists, skip
                            cross_rgbs = [imageio.imread(os.path.join(cross_dir_path, img)) for img in cross_img_path]
                            cross_rgbs_weight = np.ones((len(cross_rgbs), 3), dtype=np.float32)
                            cross_rgb = np.einsum('nhwc,nc->hwc', np.stack(cross_rgbs, axis=0), cross_rgbs_weight)
                            imageio.imwrite(os.path.join(cross_olat_sum_path, f'{hdri}.exr'), cross_rgb)
                            
                            # save normalized cross_rgb as f'{hdri}.norm.jpg'
                            cross_rgb_ldr = cv2.normalize(cross_rgb, None, 0, 255, cv2.NORM_MINMAX)
                            cross_rgb_ldr = cv2.cvtColor(cross_rgb_ldr, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(cross_olat_sum_path, f'{hdri}.norm.jpg'), cross_rgb_ldr)
                            
                        if not os.path.isfile(os.path.join(paral_olat_sum_path, f'{hdri}.exr')) or force_update: # if the hdri file already exists, skip
                            paral_rgbs = [imageio.imread(os.path.join(paral_dir_path, img)) for img in paral_img_path]
                            paral_rgbs_weight = np.ones((len(paral_rgbs), 3), dtype=np.float32)
                            paral_rgb = np.einsum('nhwc,nc->hwc', np.stack(paral_rgbs, axis=0), paral_rgbs_weight)
                            imageio.imwrite(os.path.join(paral_olat_sum_path, f'{hdri}.exr'), paral_rgb.astype(np.float32))
                            
                            # save normalized paral_rgb as f'{hdri}.norm.jpg'
                            paral_rgb_ldr = cv2.normalize(paral_rgb, None, 0, 255, cv2.NORM_MINMAX)
                            paral_rgb_ldr = cv2.cvtColor(paral_rgb_ldr, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(paral_olat_sum_path, f'{hdri}.norm.jpg'), paral_rgb_ldr)
                else:
                    if split == 'train':
                        # print(f'Skipping {row["obj"]} at cam{row["cam"]:02d} with {n_cross} cross lights and {n_paral} parallel lights, not equal lights for training.')
                        pass
                    metadata_.append(row)
            print(f'Expanded metadata from {len(metadata)} to {len(metadata_)} by adding lighting index, {expansion_counter} objects expanded.')
            metadata = metadata_
        
        self.omega_i_world = self.get_olat()
        # self.bbox_setting = self.init_bbox()
        
        self.texts = []
        self.objs = []
        self.augmented = []
        self.camera_paths = []
        self.static_paths = []
        self.static_cross_paths = []
        self.static_parallel_paths = []
        self.cross_paths = []
        self.parallel_paths = []
        self.cross_rgb_weights = [] # used to store the cross olat weights
        self.parallel_rgb_weights = [] # used to store the cross olat weights
        self.albedo_paths = []
        self.normal_paths = []
        self.specular_paths = []
        self.sigma_paths = []
        self.mask_paths = []
        self.omega_i = []
        self.windows = []
        
        print(f"Total files in LightStage dataset at {self.root_dir}: {len(metadata)}")
        for rowidx, row in enumerate(tqdm(metadata, desc='loading metadata')): # annoying when multi gpu
        # for rowidx, row in enumerate(metadata):
        
            if row['l'] <= 1 or row['l'] >= 348:
                # 2+346+2, 3,695,650 samples
                continue
            
            # general filter
            if 'fitting' not in meta_data_path:

                # task specific filter
                task = tasks[0] if len(tasks) == 1 else tasks
                if task == 'normal':
                    # when task is normal only, filter out the lighting
                    if row['l'] != 2:
                        continue
                    else:
                        pass # only pass the l==2 # verify the diffuse specular removal, 10559 samples
                else:
                    raise NotImplementedError(f'Task {task} is not implemented')
            else:
                pass

            # we use first 0.8 of the data for training, and last 0.2 for validation
            train_eval_split = 0.8
            if split == 'train':
                if rowidx / len(metadata) >= train_eval_split:
                    continue
                else:
                    pass
            elif split == 'test':
                if rowidx / len(metadata) < train_eval_split:
                # if rowidx / len(metadata) < train_eval_split or metadata[rowidx]['obj'] != 'woodball': # debug use woodball
                    continue
                else:
                    # filter out those lighting != 2 to evaluate only the static lighting
                    if metadata[rowidx]['l'] != 2: # l==0 is filtered earlier via the general filter
                        continue
                    pass

            self.texts.append(row)
            self.objs.append(row["obj"])
            self.augmented.append(row['aug'])
            
            camera_path = os.path.join(self.cam_dir, f'camera{row["cam"]:02d}.txt')
            
            if 'fitting' not in meta_data_path:
                static_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static', f'{row["i"]}_{row["j"]}.{img_ext}')
                static_cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_cross', f'{row["i"]}_{row["j"]}.{img_ext}')
                static_parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_parallel', f'{row["i"]}_{row["j"]}.{img_ext}')
                cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{img_ext}')
                parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{img_ext}')
                albedo_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'albedo', f'{row["i"]}_{row["j"]}.{img_ext}')
                normal_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'normal', f'{row["i"]}_{row["j"]}.{img_ext}')
                specular_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'specular', f'{row["i"]}_{row["j"]}.{img_ext}')
                sigma_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'sigma', f'{row["i"]}_{row["j"]}.{img_ext}')
                mask_path = ''
            else:
                static_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static.{img_ext}')
                static_cross_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_cross.{img_ext}')
                static_parallel_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_parallel.{img_ext}')
                cross_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{row["l"]:06d}.{img_ext}')
                parallel_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{row["l"]:06d}.{img_ext}')
                # cross_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_cross.{img_ext}') # hack
                # parallel_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_parallel.{img_ext}') # hack
                albedo_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'albedo.{img_ext}')
                normal_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'normal.{img_ext}')
                specular_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'specular.{img_ext}')
                sigma_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'sigma.{img_ext}')
                mask_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'mask.png')
                
            # change the cross_path and parallel_path to list of paths that consists of various lighting
            # lighting_augmentation = 'random8' # 'single', 'random2', 'random4', 'hdri'
            if self.lighting_augmentation == 'single':
                cross_path = [cross_path]
                parallel_path = [parallel_path]
                cross_rgb_weights = [(1.0, 1.0, 1.0)]
                parallel_rgb_weights = [(1.0, 1.0, 1.0)]
            elif self.lighting_augmentation == 'random8':
                random_lights = np.random.choice(self.omega_i_world.shape[0], 8, replace=False)
                random_lights = [int(x) + 2 for x in random_lights]
                cross_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                parallel_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                cross_rgb_weights = [(1.0, 1.0, 1.0)] * len(cross_path)
                parallel_rgb_weights = [(1.0, 1.0, 1.0)] * len(parallel_path)
                assert type(cross_path) == list and type(cross_path[0]) == str, f'cross_path should be a list of strings, got {type(cross_path)}'
            elif self.lighting_augmentation == 'random16':
                random_lights = np.random.choice(self.omega_i_world.shape[0], 16, replace=False)
                random_lights = [int(x) + 2 for x in random_lights]
                cross_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                parallel_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                cross_rgb_weights = [(1.0, 1.0, 1.0)] * len(cross_path)
                parallel_rgb_weights = [(1.0, 1.0, 1.0)] * len(parallel_path)
            elif self.lighting_augmentation == '-random8': # slow loading
                cross_path = []
                parallel_path = []
                for i in range(self.lighting_augmentation_pair_n):
                    random_lights = np.random.choice(self.omega_i_world.shape[0], 8, replace=False)
                    cross_path_minus = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                    parallel_path_minus = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                    cross_rgb_weights = [(1.0, 1.0, 1.0)] + [(-1.0, -1.0, -1.0)] * len(cross_path_minus)
                    parallel_rgb_weights = [(1.0, 1.0, 1.0)] + [(-1.0, -1.0, -1.0)] * len(parallel_path_minus)
                    # scale the weights element-wise to 0.25
                    w = 0.5 # recommend [0.5-0.25]
                    cross_rgb_weights = [(r*w, g*w, b*w) for (r, g, b) in cross_rgb_weights]
                    parallel_rgb_weights = [(r*w, g*w, b*w) for (r, g, b) in parallel_rgb_weights]
                    # append all white
                    cross_path.append([os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross_hdri', f'allwhite.exr')] + cross_path_minus)
                    parallel_path.append([os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel_hdri', f'allwhite.exr')] + parallel_path_minus)
            
                assert type(cross_path) == list and type(cross_path[0]) == list, f'cross_path should be a list of lists, got {type(cross_path)}'
            elif self.lighting_augmentation == '-random16':
                random_lights = np.random.choice(self.omega_i_world.shape[0], 16, replace=False)
                cross_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                parallel_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{random_light:06d}.{img_ext}') for random_light in random_lights]
                cross_rgb_weights = [(1.0, 1.0, 1.0)] + [(-1.0, -1.0, -1.0)] * len(cross_path)
                parallel_rgb_weights = [(1.0, 1.0, 1.0)] + [(-1.0, -1.0, -1.0)] * len(parallel_path)
                # scale the weights element-wise to 0.25
                w = 0.5 # recommend [0.5-0.25]
                cross_rgb_weights = [(r*w, g*w, b*w) for (r, g, b) in cross_rgb_weights]
                parallel_rgb_weights = [(r*w, g*w, b*w) for (r, g, b) in parallel_rgb_weights]
                # append all white
                cross_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross_hdri', f'allwhite.exr')] + cross_path
                parallel_path = [os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel_hdri', f'allwhite.exr')] + parallel_path
            elif self.lighting_augmentation == 'hdri':
                pass
            assert type(cross_path) == list, f'cross_path should be a list, got {type(cross_path)}'
            assert type(parallel_path) == list, f'parallel_path should be a list, got {type(parallel_path)}'
                
            # check if the paths are valid, 42k examples took 2min, remove from this and add a preprocess check
            # this only need to be enabled once, disable later to save time
            # assert os.path.isfile(camera_path), f'{camera_path} is not valid'
            # assert os.path.isfile(static_path), f'{static_path} is not valid'
            # assert os.path.isfile(static_cross_path), f'{static_cross_path} is not valid'
            # assert os.path.isfile(static_parallel_path), f'{static_parallel_path} is not valid'
            # assert os.path.isfile(cross_path), f'{cross_path} is not valid'
            # assert os.path.isfile(parallel_path), f'{parallel_path} is not valid'
            # assert os.path.isfile(albedo_path), f'{albedo_path} is not valid'
            # assert os.path.isfile(normal_path), f'{normal_path} is not valid'
            # assert os.path.isfile(specular_path), f'{specular_path} is not valid'
            # assert os.path.isfile(sigma_path), f'{sigma_path} is not valid'

            # per sample append
            self.camera_paths.append(camera_path)
            self.static_paths.append(static_path)
            self.static_cross_paths.append(static_cross_path)
            self.static_parallel_paths.append(static_parallel_path)
            self.cross_paths.append(cross_path)
            self.parallel_paths.append(parallel_path)
            self.cross_rgb_weights.append(cross_rgb_weights)
            self.parallel_rgb_weights.append(parallel_rgb_weights)
            self.albedo_paths.append(albedo_path)
            self.normal_paths.append(normal_path)
            self.specular_paths.append(specular_path)
            self.sigma_paths.append(sigma_path)
            self.mask_paths.append(mask_path)
            
            self.omega_i.append(self.omega_i_world[row['l']-2]) # 2+346+2
            # self.windows.append((row['i'], row['j'], row['res']))

        # when enable quick_val, get the first 10 samples
        if eval_first_n and split != 'train':
            
            # check if 'woodball' is in the objs, put to the first when exist
            # assert 'woodball' in self.objs, f'woodball is not in the objs: {self.objs}, this is for debugging purpose'
            # woodball_idx = self.objs.index('woodball')
            # debug_texts = [self.texts[woodball_idx], self.texts[woodball_idx+7]]
            # debug_objs = [self.objs[woodball_idx], self.objs[woodball_idx+7]]
            # debug_camera_paths = [self.camera_paths[woodball_idx], self.camera_paths[woodball_idx+7]]
            # debug_static_paths = [self.static_paths[woodball_idx], self.static_paths[woodball_idx+7]]
            # debug_static_cross_paths = [self.static_cross_paths[woodball_idx], self.static_cross_paths[woodball_idx+7]]
            # debug_static_parallel_paths = [self.static_parallel_paths[woodball_idx], self.static_parallel_paths[woodball_idx+7]]
            # debug_cross_paths = [self.cross_paths[woodball_idx], self.cross_paths[woodball_idx+7]]
            # debug_parallel_paths = [self.parallel_paths[woodball_idx], self.parallel_paths[woodball_idx+7]]
            # debug_albedo_paths = [self.albedo_paths[woodball_idx], self.albedo_paths[woodball_idx+7]]
            # debug_normal_paths = [self.normal_paths[woodball_idx], self.normal_paths[woodball_idx+7]]
            # debug_specular_paths = [self.specular_paths[woodball_idx], self.specular_paths[woodball_idx+7]]
            # debug_sigma_paths = [self.sigma_paths[woodball_idx], self.sigma_paths[woodball_idx+7]]
            
            debug_texts = []
            debug_objs = []
            debug_camera_paths = []
            debug_static_paths = []
            debug_static_cross_paths = []
            debug_static_parallel_paths = []
            debug_cross_paths = []
            debug_parallel_paths = []
            debug_albedo_paths = []
            debug_normal_paths = []
            debug_specular_paths = []
            debug_sigma_paths = []
            debug_mask_paths = []

            # get index samples with step s and crop by eval_first_n
            # eval_idx = list(range(0, len(self.texts), 64))
            n_samples_per_object = 8 # 8 samples per object, this is the number without olat
            n_sample_camera = 2
            eval_idx = list(range(0, len(self.texts), n_samples_per_object // n_sample_camera)) # 8/2 = 4, so every 4th sample, this results in 2 samples per object
            eval_idx = eval_idx[:eval_first_n]

            # truncate to eval_first_n            
            self.texts = debug_texts + [self.texts[i] for i in eval_idx]
            self.objs = debug_objs + [self.objs[i] for i in eval_idx]
            self.camera_paths = debug_camera_paths + [self.camera_paths[i] for i in eval_idx]
            self.static_paths = debug_static_paths + [self.static_paths[i] for i in eval_idx]
            self.static_cross_paths = debug_static_cross_paths + [self.static_cross_paths[i] for i in eval_idx]
            self.static_parallel_paths = debug_static_parallel_paths + [self.static_parallel_paths[i] for i in eval_idx]
            self.cross_paths = debug_cross_paths + [self.cross_paths[i] for i in eval_idx]
            self.parallel_paths = debug_parallel_paths + [self.parallel_paths[i] for i in eval_idx]
            self.albedo_paths = debug_albedo_paths + [self.albedo_paths[i] for i in eval_idx]
            self.normal_paths = debug_normal_paths + [self.normal_paths[i] for i in eval_idx]
            self.specular_paths = debug_specular_paths + [self.specular_paths[i] for i in eval_idx]
            self.sigma_paths = debug_sigma_paths + [self.sigma_paths[i] for i in eval_idx]
            self.mask_paths = debug_mask_paths + [self.mask_paths[i] for i in eval_idx]

        self.transforms = transforms.Compose(
            [
                # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
                # transforms.ToPILImage(),
            ]
        )
        self.tonemap = cv2.createTonemap(gamma=2.2)
            
    def __len__(self):
        return len(self.texts)
    
    @staticmethod
    def get_lightstage_camera(fpath, downscale=1.0):
        """
        Reads Lightstage camera parameters from file and optionally
        downscales intrinsics by a given factor.
        
        Args:
            fpath (str): Path to the Lightstage camera file.
            downscale (float): Downscaling factor, e.g., 2.0 halves the resolution.

        Returns:
            dict with keys:
                'Rt':  (3 x 4 or 4 x 4) extrinsic matrix
                'K':   (3 x 3) intrinsic matrix
                'fov': (3,) approximate field of view in degrees (for each dimension)
                'hwf': [height, width, focal_x] 
                'pp':  (2,) principal point [cx, cy]
        """
        # 1) Read lines from file
        with open(fpath) as f:
            txt = f.read().split('\n')
            
        # 2) Parse lines
        # Typically the text file has lines like:
        #   line 1: focal_x focal_y
        #   line 3: pp_x pp_y
        #   line 5: width height
        # Then lines 12..14: extrinsics
        focal = np.asarray(txt[1].split()).astype(np.float32)      # shape (2,)
        pp = np.asarray(txt[3].split()).astype(np.float32)         # shape (2,)
        resolution = np.asarray(txt[5].split()).astype(np.float32) # shape (2,) = [width, height]
        Rt = np.asarray([line.split() for line in txt[12:15]]).astype(np.float32)
        
        # 3) Compute field of view in radians, then convert to degrees
        #    fov_x = 2 * arctan( (width/2) / focal_x ), etc.
        fov = 2 * np.arctan(0.5 * resolution / focal)  # shape (2,)
        fov_deg = fov / np.pi * 180.0                  # convert to degrees
        
        # 4) If downscale != 1.0, adjust the camera intrinsics accordingly
        if downscale != 1.0:
            resolution = resolution / downscale
            focal = focal / downscale
            pp = pp / downscale
            # Recompute FOV if you want the scaled version
            # (If you keep ratio resolution/focal the same, angle stays the same,
            #  but let's re-derive it for completeness.)
            fov = 2 * np.arctan(0.5 * resolution / focal)
            fov_deg = fov / np.pi * 180.0
                
        # 5) Compose the intrinsic matrix K
        # https://stackoverflow.com/questions/74749690/how-will-the-camera-intrinsics-change-if-an-image-is-cropped-resized
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal[0]
        K[1, 1] = focal[1]
        K[0, 2] = pp[0]
        K[1, 2] = pp[1]
        
        # 6) Return a dictionary of all camera parameters
        return {
            'Rt': Rt,                       # (3 x 4) or (4 x 4) extrinsic
            'K': K,                         # (3 x 3) intrinsic
            'fov': fov_deg,                # field of view in degrees
            'hwf': [resolution[1],         # height
                    resolution[0],         # width
                    focal[0]],             # focal_x (for NeRF-style notation)
            'pp': pp                       # principal point
        }
        
    @staticmethod
    def get_olat():
                
        if socket.gethostname() == 'vgldgx01':
            olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
        elif socket.gethostname() == 'agamemnon-ub':
            olat_base = '/home/ICT2000/jyang/projects/ObjectReal/data/LSX'
            
        olat_pos_ = np.genfromtxt(f'{olat_base}/LSX3_light_positions.txt').astype(np.float32)
        olat_idx = np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
        r = R.from_euler('y', 180, degrees=True)
        olat_pos_ = (olat_pos_ @ r.as_matrix().T).astype(np.float32)
        olat_pos = olat_pos_ / (np.linalg.norm(olat_pos_, axis=-1, keepdims=True)+1e-8)
        omega_i_world = olat_pos[olat_idx-1] 
        
        return omega_i_world # (346, 3)
    
    @staticmethod
    def bbox_cropping(img, bbox):
        """
        Crop the image based on the bounding box.
        """
        # get bbox
        l, t, r, b = bbox
        return img[t:b, l:r]
    
    def __getitem__(self, idx):
        example = {}
        
        # Load data
        text = self.texts[idx]
        obj = self.objs[idx]
        augmented = self.augmented[idx]
        camera_path = self.camera_paths[idx]
        static_path = self.static_paths[idx]
        cross_path = self.cross_paths[idx]
        parallel_path = self.parallel_paths[idx]
        cross_rgb_weights = self.cross_rgb_weights[idx]
        parallel_rgb_weights = self.parallel_rgb_weights[idx]
        albedo_path = self.albedo_paths[idx]
        normal_path = self.normal_paths[idx]
        specular_path = self.specular_paths[idx]
        sigma_path = self.sigma_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # check
        example['text'] = self.texts[idx]
        example['obj_name'] = self.objs[idx]
        example['augmented'] = self.augmented[idx]
        example['obj_material'] = ''
        example['camera_path'] = self.camera_paths[idx]
        example['static_path'] = self.static_paths[idx]
        example['cross_path'] = self.cross_paths[idx]
        example['parallel_path'] = self.parallel_paths[idx]
        example['cross_rgb_weights'] = self.cross_rgb_weights[idx]
        example['parallel_rgb_weights'] = self.parallel_rgb_weights[idx]
        example['albedo_path'] = self.albedo_paths[idx]
        example['normal_path'] = self.normal_paths[idx]
        example['specular_path'] = self.specular_paths[idx]
        example['sigma_path'] = self.sigma_paths[idx]
        example['mask_path'] = self.mask_paths[idx]
        
        # get camera parameters, outgoing radiance
        cam = self.get_lightstage_camera(camera_path)
        R = cam['Rt'][:3, :3]
        t = cam['Rt'][:3, 3]
        
        # get incoming radiance
        # light_index = int(os.path.basename(cross_path).split('.')[1]) # anisotropyball/cross/cam00/cam00.000000.exr
        # if light_index in [0, 1, 348, 349]:
        #     return # skip indicators
        # if light_index > 350//2:
        #     return # skip half of the lights since most of them are black
        # if any_not_exist:
        #     print(f'Skipping {text}, olat {light_index} due to missing files, use dump_olat.py to check')
        #     return
        # omega_i = self.omega_i_world[light_index-2] # -2 to align zero-index
            
        # load image data
        static = imageio.imread(static_path)
        albedo = imageio.imread(albedo_path)
        normal = imageio.imread(normal_path)
        specular = imageio.imread(specular_path)
        sigma = imageio.imread(sigma_path)
        mask = imageio.imread(mask_path) if mask_path else (np.ones_like(static[:,:,0], dtype=np.int8) * 255) # mask is optional, use ones if not exist
        cross = []
        parallel = []
        parallel_stacked = []   
        for i in range(self.lighting_augmentation_pair_n):
            crosses = [imageio.imread(cross_path_) for cross_path_ in self.cross_paths[idx][i]]
            parallels = [imageio.imread(parallel_path_) for parallel_path_ in self.parallel_paths[idx][i]]
            cross.append(np.einsum('nhwc,nc->hwc', np.stack(crosses, axis=0), cross_rgb_weights))
            parallel.append(np.einsum('nhwc,nc->hwc', np.stack(parallels, axis=0), parallel_rgb_weights))
            if self.lighting_augmentation in ['-random8', '-random16']:
                # remove the first one as the first one is usually over exposed
                parallel_stacked.append(np.hstack(parallels[1:]) if len(parallels) > 2 else parallel)
            else:
                parallel_stacked.append(np.hstack(parallels) if len(parallels) > 1 else parallel) # for visualization purpose
        cross = np.stack(cross) if len(cross) > 1 else cross[0] # for visualization purpose
        parallel = np.stack(parallel) if len(parallel) > 1 else parallel[0]
        parallel_stacked = np.vstack(parallel_stacked) if len(parallel_stacked) > 1 else parallel_stacked[0] # for visualization purpose

        # normalize to [0,1]
        static = static if '.exr' in static_path else static / 255.0
        cross = cross if '.exr' in os.path.dirname(cross_path[0][0]) else cross / 255.0
        parallel = parallel if '.exr' in os.path.dirname(parallel_path[0][0]) else parallel / 255.0
        albedo = albedo if '.exr' in albedo_path else albedo / 255.0 * 4. # albedo is too dark
        normal = normal if '.exr' in normal_path else normal / 255.0
        specular = specular if '.exr' in specular_path else specular / 255.0
        sigma = sigma if '.exr' in sigma_path else sigma / 255.0
        mask = mask / 255.0
        
        # hdr to ldr via Apply simple Reinhard tone mapping
        # static = self.tonemap.process(static)
        # static = static.clip(0, 1)
        cross = cross.clip(0, 1)
        parallel = parallel.clip(0, 1)
        albedo = albedo.clip(0, 1)
        normal = normal.clip(-1, 1) if '.exr' in normal_path else (normal.clip(0, 1) * 2. - 1.) # normal is [-1, 1] in exr, [0, 1] in png
        specular = specular.clip(0, 1)
        sigma = sigma.clip(0, 1) / 10. # the decomposition used 10 as a clipping factor
        mask = mask.clip(0, 1)
        
        # [0,1] to [-1,1]
        static = (static - 0.5) * 2.0
        cross = (cross - 0.5) * 2.0
        parallel = (parallel - 0.5) * 2.0
        albedo = (albedo - 0.5) * 2.0

        # remove nan and inf values
        static = np.nan_to_num(static)
        cross = np.nan_to_num(cross)
        parallel = np.nan_to_num(parallel)
        albedo = np.nan_to_num(albedo)
        normal = np.nan_to_num(normal)
        specular = np.nan_to_num(specular)
        sigma = np.nan_to_num(sigma)
        mask = np.nan_to_num(mask)

        # swap x and z to align with the lotus/rgb2x
        # TODO: check rotation matrix as well
        normal[:,:,0] *= -1.
        # normal = normal[:, :, [2, 1, 0]]
        # normal_w2c = normal_w2c[:, :, [2, 1, 0]]
        
        # normal is world space normal, transform it to camera space
        normal_w2c = np.einsum('ij, hwi -> hwj', R, normal)
        
        def hdr2ldr(img):
            """
            Convert HDR image to LDR using Reinhard tone mapping.
            """
            img = img.clip(0, None)
            img = img / (img + 1.0)  # Simple Reinhard tone mapping
            img = img.clip(0, 1)  # Ensure values are in [0, 1]
            return img
        
        def normalizergb(rgb):
            """
            Normalize RGB image to [0, 1] range.
            """
            rgb = rgb.clip(0, None)
            rgb = rgb / (rgb.max() + 1e-8)
            rgb = rgb.clip(0, 1)  # Ensure values are in [0, 1]
            return rgb
        
        if 'exr' in static_path:
            # static is already in [0, 1] range, but we apply tone mapping for consistency
            static = hdr2ldr(static)
            static = normalizergb(static)  # Normalize static RGB values
        if 'exr' in cross_path[0]:
            albedo = hdr2ldr(albedo)
            albedo = normalizergb(albedo)  # Normalize albedo RGB values

        # apply transforms
        static = self.transforms(static)
        cross = torch.tensor(cross).moveaxis(3,1)
        parallel = torch.tensor(parallel).moveaxis(3,1)
        albedo = self.transforms(albedo)
        normal = self.transforms(normal)
        normal_w2c = self.transforms(normal_w2c)
        specular = self.transforms(specular)
        sigma = self.transforms(sigma)
        mask = self.transforms(mask)
        parallel_stacked = self.transforms(parallel_stacked)
        
        # get bounding box
        example['static_value'] = static
        example['cross_value'] = cross
        example['parallel_value'] = parallel
        example['albedo_value'] = albedo
        example['normal_w2c_value'] = normal_w2c
        example['normal_c2w_value'] = normal
        example['specular_value'] = specular.repeat(3, 1, 1) # repeat to 3 channels
        example['sigma_value'] = sigma
        example['mask_value'] = mask
        example['augmented'] = augmented
        example['parallel_value_hstacked'] = parallel_stacked
        
        return example
    
    
def collate_fn_lightstage(examples):
    static_pathes = [example['static_path'] for example in examples]
    cross_pathes = [example['cross_path'] for example in examples]
    parallel_pathes = [example['parallel_path'] for example in examples]
    albedo_pathes = [example['albedo_path'] for example in examples]
    normal_pathes = [example['normal_path'] for example in examples]
    specular_pathes = [example['specular_path'] for example in examples]
    sigma_pathes = [example['sigma_path'] for example in examples]
    mask_pathes = [example['mask_path'] for example in examples]

    static_values = torch.stack([example['static_value'] for example in examples])
    static_values = static_values.to(memory_format=torch.contiguous_format).float()

    cross_values = torch.stack([example['cross_value'] for example in examples])
    cross_values = cross_values.to(memory_format=torch.contiguous_format).float()

    parallel_values = torch.stack([example['parallel_value'] for example in examples])
    parallel_values = parallel_values.to(memory_format=torch.contiguous_format).float()
    
    parallel_values_hstacked = torch.stack([example['parallel_value_hstacked'] for example in examples])
    parallel_values_hstacked = parallel_values_hstacked.to(memory_format=torch.contiguous_format).float()

    albedo_values = torch.stack([example['albedo_value'] for example in examples])
    albedo_values = albedo_values.to(memory_format=torch.contiguous_format).float()

    normal_w2c_values = torch.stack([example['normal_w2c_value'] for example in examples])
    normal_w2c_values = normal_w2c_values.to(memory_format=torch.contiguous_format).float()

    normal_c2w_values = torch.stack([example['normal_c2w_value'] for example in examples])
    normal_c2w_values = normal_c2w_values.to(memory_format=torch.contiguous_format).float()

    specular_values = torch.stack([example['specular_value'] for example in examples])
    specular_values = specular_values.to(memory_format=torch.contiguous_format).float()

    sigma_values = torch.stack([example['sigma_value'] for example in examples])
    sigma_values = sigma_values.to(memory_format=torch.contiguous_format).float()
    
    mask_values = torch.stack([example['mask_value'] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    
    # get pixel values by augment
    pixel_values = torch.stack([example['static_value'] if not example['augmented'] else example['parallel_value'][0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        # values
        # "pixel_values": static_values, # hack
        "pixel_values": pixel_values, # hack
        "static_values": static_values,
        "cross_values": cross_values,
        "parallel_values": parallel_values,
        "parallel_values_hstacked": parallel_values_hstacked, # hstacked parallel values for visualization
        "albedo_values": albedo_values,
        "normal_values": normal_w2c_values, # camera space normal
        "normal_c2w_values": normal_c2w_values,
        "specular_values": specular_values,
        "sigma_values": sigma_values,
        "valid_mask_values": mask_values,
        # paths
        "static_pathes": static_pathes,
        "cross_pathes": cross_pathes,
        "parallel_pathes": parallel_pathes,
        "albedo_pathes": albedo_pathes,
        "normal_pathes": normal_pathes,
        "specular_pathes": specular_pathes,
        "sigma_pathes": sigma_pathes,
        "mask_pathes": mask_pathes,
    }