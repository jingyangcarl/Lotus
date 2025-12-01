import os
import pandas as pd
from torch.utils.data import Dataset
import imageio
import numpy as np
import socket
from scipy.spatial.transform import Rotation as _R
import torch
import json
from tqdm.rich import tqdm
from torchvision import transforms
import cv2

class ObjaverseTransform:
    pass

class ObjaverseDataset(Dataset):

    def __init__(self, split='train', tasks='', ori_aug_ratio='1:1', lighting_aug='random8', lighting_aug_pair_n = 2, eval_first_n=None, img_ext='jpg', n_rotations=1):

        assert split in ['train', 'test'], f'Invalid split: {split}'
        
        v = 'v1.3'
        self.root_dir = '/labworking/Users_A-L/jyang/data/LightStageObjectDB'
        self.root_dir = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_dir' # local cache, no IO bottle neck
        self.root_dir = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all'
        # self.root_dir = '/home/jyang/projects/dataCollectionObjaverse/renderings/output_all_v2'
        self.img_ext = img_ext # 'exr' or 'jpg' # TODO: jpg need to updated to compatible with negative values, running now
        # when use exr, the evaluation results may look different to the validation due to 
        meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_fitting_512_ck.csv'
        # The code `self.dataset_dir` is accessing the `dataset_dir` attribute of the current object
        # (instance) in Python. This code is typically found within a class definition where `self`
        # refers to the current instance of the class. By accessing `self.dataset_dir`, the code is
        # retrieving the value stored in the `dataset_dir` attribute of the current object.
        self.dataset_dir = f'{self.root_dir}/renderings/'
        self.cam_dir = f'/home/jyang/data/LightStageObjectDB/Redline/exr/{v}/{v}_2/cameras'
        self.hdri_dir = '/home/jyang/data/lightProbe/general/jpg'
        
        self.original_augmentation_ratio = ori_aug_ratio
        self.lighting_augmentation = lighting_aug
        self.lighting_augmentation_pair_n = lighting_aug_pair_n # number of pairs to generate for lighting augmentation, aiming on the same prediction
        self.n_rotations = n_rotations # rotationo samples
        
        # load json file
        metadata = []   
        objects = list(os.scandir(self.dataset_dir))
        for obj in objects:
            row = {}
            row['obj'] = obj.name
            row['cam'] = 0
            row['mat'] = ''
            row['l'] = 0
            row['res'] = 512
            metadata.append(row)
                
        # add a manual expansion here since the cropping's under processing
        if 'fitting' in meta_data_path:
            metadata_ = []
            expansion_counter = 0
            for row in tqdm(metadata, 'expanding metadata'):
                cross_dir_path = os.path.join(self.dataset_dir, row["obj"], 'lighting')
                paral_dir_path = os.path.join(self.dataset_dir, row["obj"], 'lighting')
                
                cross_img_path = sorted(os.listdir(cross_dir_path)) if os.path.isdir(cross_dir_path) else []
                paral_img_path = sorted(os.listdir(paral_dir_path)) if os.path.isdir(paral_dir_path) else []
                
                cross_img_path = [p for p in cross_img_path if 'olat_' in p]
                paral_img_path = [p for p in paral_img_path if 'olat_' in p]
                
                n_cross = len(cross_img_path)
                n_paral = len(paral_img_path)
                
                if n_cross == n_paral:
                    # for l in range(350):
                    for l in range(10): # debug #
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
                            metadata_.append(row_) # double the samples to keep the same number of samples as 1:1
                        elif self.original_augmentation_ratio == '0:1':
                            row_ = row.copy()
                            row_['aug'] = True
                            metadata_.append(row_)
                            metadata_.append(row_)
                        else:
                            raise NotImplementedError(f'Original augmentation ratio {ori_aug_ratio} is not supported')
                        
                    expansion_counter += 1
                    
                    # add a code to add all the olat images together here to make the -random8 faster, this only need to be done once
                    # cross_olat_sum_path = cross_dir_path.replace('cross', 'cross_hdri')
                    # paral_olat_sum_path = paral_dir_path.replace('parallel', 'parallel_hdri')
                    # os.makedirs(cross_olat_sum_path, exist_ok=True)
                    # os.makedirs(paral_olat_sum_path, exist_ok=True)
                    
                    # hdri_list = [
                    #     'all_white',
                    #     "city",
                    #     "night"
                    # ]
                    
                    # for hdri in hdri_list:
                    #     force_update = False
                    #     if not os.path.isfile(os.path.join(cross_olat_sum_path, f'{hdri}.exr')) or force_update: # if the hdri file already exists, skip
                    #         cross_rgbs = [imageio.imread(os.path.join(cross_dir_path, img)) for img in cross_img_path]
                    #         cross_rgbs_weight = np.ones((len(cross_rgbs), 3), dtype=np.float32)
                    #         cross_rgb = np.einsum('nhwc,nc->hwc', np.stack(cross_rgbs, axis=0), cross_rgbs_weight)
                    #         imageio.imwrite(os.path.join(cross_olat_sum_path, f'{hdri}.exr'), cross_rgb)
                            
                    #         # save normalized cross_rgb as f'{hdri}.norm.jpg'
                    #         cross_rgb_ldr = cv2.normalize(cross_rgb, None, 0, 255, cv2.NORM_MINMAX)
                    #         cross_rgb_ldr = cv2.cvtColor(cross_rgb_ldr, cv2.COLOR_RGB2BGR)
                    #         cv2.imwrite(os.path.join(cross_olat_sum_path, f'{hdri}.norm.jpg'), cross_rgb_ldr)
                            
                    #     if not os.path.isfile(os.path.join(paral_olat_sum_path, f'{hdri}.exr')) or force_update: # if the hdri file already exists, skip
                    #         paral_rgbs = [imageio.imread(os.path.join(paral_dir_path, img)) for img in paral_img_path]
                    #         paral_rgbs_weight = np.ones((len(paral_rgbs), 3), dtype=np.float32)
                    #         paral_rgb = np.einsum('nhwc,nc->hwc', np.stack(paral_rgbs, axis=0), paral_rgbs_weight)
                    #         imageio.imwrite(os.path.join(paral_olat_sum_path, f'{hdri}.exr'), paral_rgb.astype(np.float32))
                            
                    #         # save normalized paral_rgb as f'{hdri}.norm.jpg'
                    #         paral_rgb_ldr = cv2.normalize(paral_rgb, None, 0, 255, cv2.NORM_MINMAX)
                    #         paral_rgb_ldr = cv2.cvtColor(paral_rgb_ldr, cv2.COLOR_RGB2BGR)
                    #         cv2.imwrite(os.path.join(paral_olat_sum_path, f'{hdri}.norm.jpg'), paral_rgb_ldr)
                else:
                    if split == 'train':
                        # print(f'Skipping {row["obj"]} at cam{row["cam"]:02d} with {n_cross} cross lights and {n_paral} parallel lights, not equal lights for training.')
                        pass
                    metadata_.append(row)
            print(f'Expanded metadata from {len(metadata)} to {len(metadata_)} by adding lighting index, {expansion_counter} objects expanded.')
            metadata = metadata_
        
        self.omega_i_world = self.get_olat() # all olat lighting direction
        self.hdri_in_olats, self.hdri_paths = self.get_hdri_in_olats(self.hdri_dir, first_n=41, n_rot=self.n_rotations) # precompute hdri to olat mapping
        # self.bbox_setting = self.init_bbox()
        
        # rotate self.omega_i_world around z for 90 degree to cancel the blender rotaiton
        r = _R.from_euler('x', 90, degrees=True)
        self.omega_i_world = (self.omega_i_world @ r.as_matrix().T).astype(np.float32)

        self.metas = []
        self.objs = []
        self.objs_mat = []
        self.objs_des = []
        self.augmented = []
        self.camera_paths = []
        self.static_paths = []
        self.static_cross_paths = []
        self.static_parallel_paths = []
        self.olat_cross_paths = []
        self.olat_parallel_paths = []
        self.olat_wi_dirs = []
        self.olat_wi_rgbs = []
        self.albedo_paths = []
        self.normal_paths = []
        self.specular_paths = []
        self.sigma_paths = []
        self.mask_paths = []
        self.windows = []
        
        print(f"Total files in Objaverse dataset at {self.root_dir}: {len(metadata)}")
        for rowidx, row in enumerate(tqdm(metadata, desc='loading metadata')): # annoying when multi gpu
        # for rowidx, row in enumerate(metadata):
        
            # clean up the metadata
            metadata[rowidx]['mat'] = '' if pd.isna(metadata[rowidx]['mat']) else metadata[rowidx]['mat']
            metadata[rowidx]['des'] = ''
        
            # general filter to remove the first 2 and last 2 lighting, since they are not OLAT sequence
            if row['l'] <= 1-2 or row['l'] >= 348-2:
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

            # we use first 0.8 (200*0.8=160) of the data for training, and last 0.2 (200*0.2=40) for validation
            train_eval_split = 0
            if split == 'train':
                if rowidx / len(metadata) >= train_eval_split:
                    continue
                else:
                    pass
            elif split == 'test':
                if rowidx / len(metadata) < train_eval_split: # add a gap of 0.2 for debugging the training and testing
                # if rowidx / len(metadata) < train_eval_split or metadata[rowidx]['obj'] != 'woodball': # debug use woodball
                    continue
                else:
                    # filter out those lighting != 2 to evaluate only the static lighting
                    if metadata[rowidx]['l'] != 2: # l==0 is filtered earlier via the general filter
                        continue

                    if rowidx / len(metadata) < train_eval_split:
                        metadata[rowidx]['des'] += '+' # add a tag to distinguish the training and testing object
                    pass

            self.metas.append(row)
            self.objs.append(row["obj"])
            self.objs_mat.append(row['mat'])
            self.objs_des.append(row['des'])
            self.augmented.append(row['aug'])
            
            camera_path = os.path.join(self.cam_dir, f'camera{row["cam"]:02d}.txt')
            
            if 'fitting' not in meta_data_path:
                # static_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # static_cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_cross', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # static_parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_parallel', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # olat_cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{self.img_ext}')
                # olat_parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{self.img_ext}')
                # albedo_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'albedo', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # normal_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'normal', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # specular_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'specular', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # sigma_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'sigma', f'{row["i"]}_{row["j"]}.{self.img_ext}')
                # mask_path = ''
                pass
            else:
                static_path = os.path.join(self.dataset_dir, row["obj"], 'lighting', 'all_white', f'frame_{row["cam"]+1:04d}.png')
                static_cross_path = static_path
                static_parallel_path = static_path
                olat_cross_path = os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', f'frame_{row["cam"]+1:04d}.png')
                olat_parallel_path = olat_cross_path
                albedo_path = os.path.join(self.dataset_dir, row["obj"], 'gbuffers', 'albedo', f'Image{row["cam"]+1:04d}.exr')
                normal_path = os.path.join(self.dataset_dir, row["obj"], 'gbuffers', 'normal', f'Image{row["cam"]+1:04d}.exr')
                specular_path = os.path.join(self.dataset_dir, row["obj"], 'gbuffers', 'specular', f'Image{row["cam"]+1:04d}.exr')
                sigma_path = os.path.join(self.dataset_dir, row["obj"], 'gbuffers', 'specular', f'Image{row["cam"]+1:04d}.exr')
                mask_path = os.path.join(self.dataset_dir, row["obj"], 'gbuffers', 'mask', f'Image{row["cam"]+1:04d}.exr')

            # change the cross_path and parallel_path to list of paths that consists of various lighting
            # lighting_augmentation = 'random8' # 'single', 'random2', 'random4', 'hdri'
            
            def get_random_olat(n=2, n_olat_compose=1, olat_wi_rgb_intensity=1.0):
                
                # the weights should be scaled by 1/N_OLATS to keep the same total intensity
                # this could lead to extreme dimming when we only pick randomly a few lights, we therefore use a intensity_scalar to compensate
                # therefore, the intensity weights will be consistent when sample different number of lights
                N_OLATS = self.omega_i_world.shape[0]
                
                # prepare the return variables
                olat_cross_path = []
                olat_parallel_path = []
                olat_wi_rgb = []
                olat_wi_dir = []
                
                if n_olat_compose > 0:
                    # sample n random lights from self.omega_i_world
                    for i in range(n):
                        # sample lights and weights and dir
                        olat_random_lights = np.random.choice(self.omega_i_world.shape[0], n_olat_compose, replace=False)
                        olat_wi_dir.append([self.omega_i_world[random_light] for random_light in olat_random_lights])
                        olat_wi_rgb.append(olat_wi_rgb_intensity * np.ones((n_olat_compose, 3), dtype=np.float32) / N_OLATS) # use ones as the rgb weight
                        # get corresponding paths
                        # olat_random_lights = [int(x) + 2 for x in olat_random_lights] # map this actual light index to the light index in the file name
                        olat_cross_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', f'frame_{row["cam"]+1:04d}.png') for random_light in olat_random_lights])
                        olat_parallel_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', f'frame_{row["cam"]+1:04d}.png') for random_light in olat_random_lights])
                    assert type(olat_cross_path) == list and type(olat_cross_path[0]) == list and type(olat_cross_path[0][0] == str), f'cross_path should be a list of strings, got {type(olat_cross_path)}'
                    return olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb
                
                elif n_olat_compose < 0:
                    raise NotImplementedError('Negative n random olat not implemented yet')
                    # sample n random lights from self.omega_i_world and subtract from all lights
                    n_olat_compose = -n_olat_compose
                    # itsty_scale = 0.25 # to avoid overexposure
                    for i in range(n):
                        # sample lights and weights and dir
                        olat_random_lights = np.random.choice(self.omega_i_world.shape[0], n_olat_compose, replace=False)
                        olat_omega_i_dir_minus = [self.omega_i_world[random_light] for random_light in olat_random_lights]
                        olat_wi_dir.append(np.vstack((np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array(olat_omega_i_dir_minus)))) # add one for all white
                        olat_omega_i_rgb_minus = -np.ones((n_olat_compose, 3), dtype=np.float32) / N_OLATS # use ones as the rgb weight
                        olat_wi_rgb.append(olat_wi_rgb_intensity * np.vstack((np.ones((1, 3), dtype=np.float32), olat_omega_i_rgb_minus))) # add one for all white
                        # get corresponding paths
                        olat_random_lights = [int(x) + 2 for x in olat_random_lights] # map this actual light index to the light index in the file name
                        olat_cross_path_minus = [os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', 'cross', f'{random_light:06d}.{self.img_ext}') for random_light in olat_random_lights]
                        olat_parallel_path_minus = [os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', 'parallel', f'{random_light:06d}.{self.img_ext}') for random_light in olat_random_lights]
                        olat_cross_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', 'cross_hdri', f'allwhite.exr')] + olat_cross_path_minus)
                        olat_parallel_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{row["l"]}', 'parallel_hdri', f'allwhite.exr')] + olat_parallel_path_minus)
                    assert type(olat_cross_path) == list and type(olat_cross_path[0]) == list and type(olat_cross_path[0][0] == str), f'cross_path should be a list of strings, got {type(olat_cross_path)}'
                    return olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb
                
            def get_fixed_olat(n=20, n_olat_compose=1, olat_wi_rgb_intensity=1.0):
                
                N_OLATS = self.omega_i_world.shape[0]
                olat_cross_path = []
                olat_parallel_path = []
                olat_wi_rgb = []
                olat_wi_dir = []
                
                if n > 0:
                    # intensity_scalar = self.olat_wi_rgb_intensity['random1']
                    # olat_img_scalar = self.olat_img_intensity['random1']
                    olat_step = N_OLATS // n
                    for i in range(0, N_OLATS//2, olat_step//2):
                        olat_wi_dir.append([self.omega_i_world[i+j] for j in range(n_olat_compose)])
                        olat_wi_rgb.append(olat_wi_rgb_intensity * np.ones((1, 3), dtype=np.float32) / N_OLATS) # use ones as the rgb weight
                        olat_cross_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{i+j}', f'frame_{row["cam"]+1:04d}.png') for j in range(n_olat_compose)])
                        olat_parallel_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{i+j}', f'frame_{row["cam"]+1:04d}.png') for j in range(n_olat_compose)])
                    assert type(olat_cross_path) == list and type(olat_cross_path[0]) == list and type(olat_cross_path[0][0]) == str, f'cross_path should be a list of strings, got {type(olat_cross_path)}'
                    return olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb
                    
            def get_hdri(n=20, use_n_olat=20, olat_wi_rgb_intensity=1.0, mode='random'):
                
                N_OLATS = self.omega_i_world.shape[0]
                olat_cross_path = []
                olat_parallel_path = []
                olat_wi_rgb = []
                olat_wi_dir = []
                
                if mode == 'fixed':
                    hdri_indices = np.linspace(0, len(self.hdri_in_olats)-1, n*self.n_rotations, dtype=np.int32)
                else:
                    hdri_indices = np.random.choice(len(self.hdri_in_olats), n*self.n_rotations, replace=False)
                
                for hdri_idx in hdri_indices:
                    hdri_L, hdri_rgb = self.hdri_in_olats[hdri_idx]
                    olat_selected = np.linspace(0, N_OLATS-1, use_n_olat, dtype=np.int32)
                    hdri_ls = hdri_L[olat_selected]
                    hdri_rgb = hdri_rgb[olat_selected]
                    olat_wi_dir.append(hdri_ls)
                    olat_wi_rgb.append(olat_wi_rgb_intensity * hdri_rgb / N_OLATS) # use ones as the rgb weight
                    olat_cross_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{olat_idx}', f'frame_{row["cam"]+1:04d}.png') for olat_idx in olat_selected])
                    olat_parallel_path.append([os.path.join(self.dataset_dir, row["obj"], 'lighting', f'olat_{olat_idx}', f'frame_{row["cam"]+1:04d}.png') for olat_idx in olat_selected])
                    assert type(olat_cross_path) == list and type(olat_cross_path[0]) == list and type(olat_cross_path[0][0]) == str, f'cross_path should be a list of strings, got {type(olat_cross_path)}'
                
                return olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb
            
        
            self.olat_wi_rgb_intensity = {
                'random1': 100.0,
                'random2': 100.0,
                'random8': 100.0,
                'random16': 50.0,
                'fixed4_via1': 100.0,
                'fixed20_via1': 100.0,
                'fixed40_via1': 100.0,
                'hdri4_olat20': 100.0,
                'hdri20_olat20': 100.0,
                'hdri40_olat20': 100.0,
            } # this factor controls the light source brightness, the higher the brighter
            self.olat_img_intensity = {
                'random1': 80.0,
                'random2': 40.0,
                'random8': 20.0,
                'random16': 5.0,
                'fixed4_via1': 15.0,
                'fixed20_via1': 15.0,
                'fixed40_via1': 15.0,
                'hdri4_olat20': 10.0,
                'hdri20_olat20': 10.0,
                'hdri40_olat20': 10.0,
            } # this factor controls the image brightness, the higher the brighter
            if self.lighting_augmentation.startswith('random') and '_' not in self.lighting_augmentation:
                n_olat_compose = int(self.lighting_augmentation.replace('random', ''))
                n_target_olat = self.lighting_augmentation_pair_n
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_random_olat(n_target_olat, n_olat_compose, self.olat_wi_rgb_intensity[self.lighting_augmentation])
            elif self.lighting_augmentation == '-random8': 
                n_target_olat = self.lighting_augmentation_pair_n
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_random_olat(n_target_olat, -8, 0.25)
            elif self.lighting_augmentation == '-random16':
                n_target_olat = self.lighting_augmentation_pair_n
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_random_olat(n_target_olat, -16)
            elif self.lighting_augmentation.startswith('fixed'): # fixed20_via1
                # fixed20_via1, generate fixed 20 olats using single olat each time
                n_target_olat, n_olat_compose = self.lighting_augmentation.split('_')
                n_target_olat = int(n_target_olat.replace('fixed', ''))
                n_olat_compose = int(n_olat_compose.replace('via', ''))
                assert n_olat_compose == 1, 'fixed only support via1 for now'
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_fixed_olat(n_target_olat, n_olat_compose, self.olat_wi_rgb_intensity[self.lighting_augmentation])
            elif self.lighting_augmentation.startswith('hdri'): 
                # hdri4_olat20, generate 4 hdris using 20 olats each
                n_target_hdri, n_olat_compose = self.lighting_augmentation.split('_')
                n_target_hdri = int(n_target_hdri.replace('hdri', ''))
                n_olat_compose = int(n_olat_compose.replace('olat', ''))
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_hdri(n_target_hdri, n_olat_compose, self.olat_wi_rgb_intensity[self.lighting_augmentation], mode='fixed')
            elif self.lighting_augmentation == 'random1+hdri_olat20':
                n_olat_compose, n_olat_compose_ = self.lighting_augmentation.split('_')
                
                n_olat_compose = int(n_olat_compose.replace('random', ''))
                n_target_olat = self.lighting_augmentation_pair_n // 2
                olat_cross_path, olat_parallel_path, olat_wi_dir, olat_wi_rgb = get_random_olat(n_target_olat, n_olat_compose, self.olat_wi_rgb_intensity[self.lighting_augmentation])

                n_target_hdri = self.lighting_augmentation_pair_n - n_target_olat
                n_olat_compose = int(n_olat_compose_.replace('olat', ''))
                olat_cross_path_, olat_parallel_path_, olat_wi_dir_, olat_wi_rgb_ = get_hdri(n_target_hdri, 20, self.olat_wi_rgb_intensity[self.lighting_augmentation])
                
                
            else:
                raise NotImplementedError(f'Lighting augmentation {self.lighting_augmentation} is not implemented')
            assert type(olat_cross_path) == list, f'cross_path should be a list, got {type(olat_cross_path)}'
            assert type(olat_parallel_path) == list, f'parallel_path should be a list, got {type(olat_parallel_path)}'
            assert type(olat_wi_dir) == list, f'olat_omega_i should be a list, got {type(olat_wi_dir)}'
            assert type(olat_wi_rgb) == list, f'olat_omega_i_rgb should be a list, got {type(olat_wi_rgb)}'
                
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
            self.olat_cross_paths.append(olat_cross_path)
            self.olat_parallel_paths.append(olat_parallel_path)
            self.olat_wi_dirs.append(olat_wi_dir)
            self.olat_wi_rgbs.append(olat_wi_rgb)
            self.albedo_paths.append(albedo_path)
            self.normal_paths.append(normal_path)
            self.specular_paths.append(specular_path)
            self.sigma_paths.append(sigma_path)
            self.mask_paths.append(mask_path)
            

        # when enable quick_val, get the first 10 samples
        if eval_first_n and split == 'test':
            
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
            
            debug_metas = []
            debug_objs = []
            debug_objs_mat = []
            debug_objs_des = []
            debug_camera_paths = []
            debug_static_paths = []
            debug_static_cross_paths = []
            debug_static_parallel_paths = []
            debug_olat_cross_paths = []
            debug_olat_parallel_paths = []
            debug_olat_wi_dirs = []
            debug_olat_wi_rgbs = []
            debug_albedo_paths = []
            debug_normal_paths = []
            debug_specular_paths = []
            debug_sigma_paths = []
            debug_mask_paths = []

            # filter out the samples and crop by eval_first_n
            assert len(self.metas) == len(self.objs) == len(self.camera_paths) == len(self.static_paths) == len(self.static_cross_paths) == len(self.static_parallel_paths) == len(self.olat_cross_paths) == len(self.olat_parallel_paths) == len(self.albedo_paths) == len(self.normal_paths) == len(self.specular_paths) == len(self.sigma_paths) == len(self.mask_paths), 'Length of texts, objs, camera_paths, static_paths, cross_paths, parallel_paths, albedo_paths, normal_paths, specular_paths, sigma_paths, mask_paths are not equal'
            cam_list = [0]
            eval_idx = []
            for i, x in enumerate(self.metas):
                if int(x['cam']) in cam_list and x['aug'] == False:
                    eval_idx.append(i)
            eval_idx = eval_idx[:eval_first_n] if eval_first_n < len(eval_idx) else eval_idx
            print(f'Evaluation: {split} set length is truncated from {len(self.metas)} to {len(eval_idx)}')

            # truncate to eval_first_n            
            self.metas = debug_metas + [self.metas[i] for i in eval_idx]
            self.objs = debug_objs + [self.objs[i] for i in eval_idx]
            self.objs_mat = debug_objs_mat + [self.objs_mat[i] for i in eval_idx]
            self.objs_des = debug_objs_des + [self.objs_des[i] for i in eval_idx]
            self.camera_paths = debug_camera_paths + [self.camera_paths[i] for i in eval_idx]
            self.static_paths = debug_static_paths + [self.static_paths[i] for i in eval_idx]
            self.static_cross_paths = debug_static_cross_paths + [self.static_cross_paths[i] for i in eval_idx]
            self.static_parallel_paths = debug_static_parallel_paths + [self.static_parallel_paths[i] for i in eval_idx]
            self.olat_cross_paths = debug_olat_cross_paths + [self.olat_cross_paths[i] for i in eval_idx]
            self.olat_parallel_paths = debug_olat_parallel_paths + [self.olat_parallel_paths[i] for i in eval_idx]
            self.olat_wi_dirs = debug_olat_wi_dirs + [self.olat_wi_dirs[i] for i in eval_idx]
            self.olat_wi_rgbs = debug_olat_wi_rgbs + [self.olat_wi_rgbs[i] for i in eval_idx]
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
        return len(self.metas)
    
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
    def get_lightstage_config():
        """This function returns the path to the Lightstage configuration file."""
        
        if socket.gethostname() == 'vgldgx01':
            olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
        elif socket.gethostname() == 'agamemnon-ub':
            olat_base = '/home/ICT2000/jyang/projects/ObjectReal/data/LSX'
            
        meta = {
            'pos_main': np.loadtxt(os.path.join(olat_base, 'LSX3_light_positions.txt')),
            'pos_daughter': np.loadtxt(os.path.join(olat_base, 'LSX3_light_only_daughter_positions.txt')),
            'pos_main_w_daughter': np.loadtxt(os.path.join(olat_base, 'LSX3_light_w_daughter_positions.txt')),
            'mapping_main': np.loadtxt(os.path.join(olat_base, 'Light_Probe_Mapping_Main_WO_Daughter.txt')),
            'mapping_daughter': np.loadtxt(os.path.join(olat_base, 'Light_Probe_Mapping_Main_Only_Daughter.txt')),
            'mapping_main_w_daughter': np.loadtxt(os.path.join(olat_base, 'Light_Probe_Mapping_Main_W_Daughter.txt')),
            'z_spiral': np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
        }
            
        return olat_base, meta

    @staticmethod
    def get_olat():
                
        if socket.gethostname() == 'vgldgx01':
            olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
        elif socket.gethostname() == 'agamemnon-ub':
            olat_base = '/home/ICT2000/jyang/projects/ObjectReal/data/LSX'
            
        olat_pos_ = np.genfromtxt(f'{olat_base}/LSX3_light_positions.txt').astype(np.float32)
        olat_idx = np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
        r = _R.from_euler('y', 180, degrees=True)
        olat_pos_ = (olat_pos_ @ r.as_matrix().T).astype(np.float32)
        olat_pos = olat_pos_ / (np.linalg.norm(olat_pos_, axis=-1, keepdims=True)+1e-8)
        omega_i_world = olat_pos[olat_idx-1] 
        
        return omega_i_world # (346, 3)
    
    @staticmethod
    def get_hdri_in_olats(hdri_root, first_n=None, n_rot=1):
        
        if socket.gethostname() == 'vgldgx01':
            olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
        elif socket.gethostname() == 'agamemnon-ub':
            olat_base = '/home/ICT2000/jyang/projects/ObjectReal/data/LSX'
            
        hdri_items = sorted(os.listdir(hdri_root))
        hdri_items = hdri_items[:first_n] if first_n is not None else hdri_items
        olat_idx = np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
        hdris = []
        hdri_paths = []
        for hdri_item in tqdm(hdri_items):
            hdri_path = os.path.join(hdri_root, hdri_item)
            hdri_paths.append(hdri_path)
            hdri = imageio.imread(hdri_path) / 255.0
            
            for rot in range(0, 360-1, 360//n_rot):
                hdri = np.roll(hdri, shift=rot, axis=1)
                hdri_L, hdri_rgb = ObjaverseDataset.hdri_to_olats(hdri=hdri) # precompute the olat weights for this hdri
                hdris.append((hdri_L[olat_idx-1], hdri_rgb[olat_idx-1]))

        return hdris, hdri_paths
    
    @staticmethod
    def get_olat_hdri(olat_idx=0, sampling_mode='main', h=256, w=512):
        """This function returns a HDRI map of the given OLAT index."""

        olat_base, meta = ObjaverseDataset.get_lightstage_config()
        hdri = np.zeros((h, w, 3), dtype=np.float32)
        
        if sampling_mode == 'main':
            # 346 regions
            L, lsx_mapping = meta['pos_main'], meta['mapping_main']
        elif sampling_mode == 'daughter':
            # 346 * 6 regions
            L, lsx_mapping = meta['pos_daughter'], meta['mapping_daughter']
        elif sampling_mode == 'main_w_daughter':
            # 346 * 7 regions
            L, lsx_mapping = meta['pos_main_w_daughter'], meta['mapping_main_w_daughter']
            
        # location normalization
        L = L / (np.linalg.norm(L, axis=-1, keepdims=True)+1e-8)
        
        # calculate average color in the region
        mapping = cv2.resize(lsx_mapping.reshape((256, 512)), hdri.shape[:2][::-1]) # raise size to match hdri
        hdri[mapping == meta['z_spiral'][olat_idx+1]] = 1.0 # set the region to white

        return hdri

    @staticmethod
    def hdri_to_olats(sampling_mode='main', hdri=None):
    
        Ls, rgbs = [], []
        olat_base, meta = ObjaverseDataset.get_lightstage_config()
            
        if sampling_mode == 'main':
            # 346 regions
            L, lsx_mapping = meta['pos_main'], meta['mapping_main']
        elif sampling_mode == 'daughter':
            # 346 * 6 regions
            L, lsx_mapping = meta['pos_daughter'], meta['mapping_daughter']
        elif sampling_mode == 'main_w_daughter':
            # 346 * 7 regions
            L, lsx_mapping = meta['pos_main_w_daughter'], meta['mapping_main_w_daughter']
        
        # location normalization
        L = L / (np.linalg.norm(L, axis=-1, keepdims=True)+1e-8)
        
        r = _R.from_euler('y', 180, degrees=True)
        L = (L @ r.as_matrix().T).astype(np.float32)
        L = L / (np.linalg.norm(L, axis=-1, keepdims=True)+1e-8)
        
        # calculate average color in the region
        mapping = cv2.resize(lsx_mapping.reshape((256, 512)), hdri.shape[:2][::-1]) # raise size to match hdri
        stack = np.dstack((hdri, mapping.astype(np.uint))).reshape((-1,4)) # append idx as forth channel for further sorting
        stack = stack[stack[:, -1].argsort()] # sorth pixels by idx channel
        groups = np.split(stack[:,:3], np.unique(stack[:,-1], return_index=True)[1][1:]) # group all pixels by values
        rgb = np.array([l.mean(axis=0) for l in groups])
        
        Ls.append(L)
        rgbs.append(rgb)
        
        Ls = np.vstack(Ls)
        rgbs = np.vstack(rgbs)

        return Ls, rgbs
    
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
        text = self.metas[idx]
        obj = self.objs[idx]
        augmented = self.augmented[idx]
        camera_path = self.camera_paths[idx]
        static_path = self.static_paths[idx]
        static_cross_path = self.static_cross_paths[idx]
        static_parallel_path = self.static_parallel_paths[idx]
        olat_cross_paths = self.olat_cross_paths[idx]
        olat_parallel_paths = self.olat_parallel_paths[idx]
        olat_omega_i_dirs = self.olat_wi_dirs[idx]
        olat_omega_i_rgbs = self.olat_wi_rgbs[idx]
        albedo_path = self.albedo_paths[idx]
        normal_path = self.normal_paths[idx]
        specular_path = self.specular_paths[idx]
        sigma_path = self.sigma_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # check
        example['meta'] = self.metas[idx]
        example['obj_name'] = self.objs[idx]
        example['augmented'] = self.augmented[idx]
        example['obj_material'] = self.objs_mat[idx]
        example['obj_description'] = self.objs_des[idx]
        example['camera_path'] = self.camera_paths[idx]
        example['static_path'] = self.static_paths[idx]
        example['static_cross_path'] = self.static_cross_paths[idx]
        example['static_parallel_path'] = self.static_parallel_paths[idx]
        example['cross_path'] = self.olat_cross_paths[idx]
        example['parallel_path'] = self.olat_parallel_paths[idx]
        example['olat_omega_i_dirs'] = self.olat_wi_dirs[idx]
        example['olat_omega_i_rgbs'] = self.olat_wi_rgbs[idx]
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
        static = imageio.imread(static_path)[...,:3]
        static_cross = imageio.imread(static_cross_path)[...,:3]
        static_parallel = imageio.imread(static_parallel_path)[...,:3]
        albedo = imageio.imread(albedo_path)[...,:3]
        normal_c2w = imageio.imread(normal_path)[...,:3]
        normal_w2c = imageio.imread(normal_path.replace('output_all', 'output_all_v2_1frame_1lighting').replace('/normal/', '/normal_w2c/'))[...,:3]
        specular = imageio.imread(specular_path)[...,:1]
        sigma = imageio.imread(sigma_path)[...,:3]
        mask = albedo[...,-1] if mask_path else (np.ones_like(static[:,:,0], dtype=np.int8) * 255) # mask is optional, use ones if not exist
        
        # resize to 512x512
        # static = cv2.resize(static, (512, 512), interpolation=cv2.INTER_LINEAR)
        # static_cross = cv2.resize(static_cross, (512, 512), interpolation=cv2.INTER_LINEAR)
        # static_parallel = cv2.resize(static_parallel, (512, 512), interpolation=cv2.INTER_LINEAR)
        # albedo = cv2.resize(albedo, (512, 512), interpolation=cv2.INTER_LINEAR)
        # normal_c2w = cv2.resize(normal_c2w, (512, 512), interpolation=cv2.INTER_LINEAR)
        # normal_w2c = cv2.resize(normal_w2c, (512, 512), interpolation=cv2.INTER_LINEAR)
        # specular = cv2.resize(specular, (512, 512), interpolation=cv2.INTER_LINEAR)
        # sigma = cv2.resize(sigma, (512, 512), interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # normalize to [0,1], except normal which is [-1,1]
        static = static if '.exr' in static_path else static / 255.0
        static_cross = static_cross if '.exr' in static_cross_path else static_cross / 255.0
        static_parallel = static_parallel if '.exr' in static_parallel_path else static_parallel / 255.0
        albedo = albedo if '.exr' in albedo_path else albedo / 255.0 * np.pi # albedo is too dark
        normal_c2w = normal_c2w if '.exr' in normal_path else (normal_c2w / 255.0) * 2.0 - 1.0 # normal is [0, 1] in png, convert to [-1, 1]
        normal_w2c = normal_w2c if '.exr' in normal_path else (normal_c2w / 255.0) * 2.0 - 1.0 
        specular = specular if '.exr' in specular_path else specular / 255.0 * np.pi * 2.0 # specular is too dark
        sigma = sigma if '.exr' in sigma_path else sigma / 255.0
        mask = mask / 1.0

        normal_c2w = normal_c2w / (np.linalg.norm(normal_c2w, axis=-1, keepdims=True) + 1e-8) # renormalize
        normal_w2c = normal_w2c / (np.linalg.norm(normal_w2c, axis=-1, keepdims=True) + 1e-8) # renormalize
        
        # get olat_cross
        cross = []
        parallel = []
        irradiance = []
        parallel_stacked = []   
        for i in range(len(olat_cross_paths)):
            olat_img_scalar = self.olat_img_intensity[self.lighting_augmentation]
            crosses = [imageio.imread(cross_path_)[...,:3] for cross_path_ in olat_cross_paths[i]]
            parallels = [imageio.imread(parallel_path_)[...,:3] for parallel_path_ in olat_parallel_paths[i]]
            crosses = [cross if '.exr' in olat_cross_paths[i][j] else cross / 255.0 * olat_img_scalar for j, cross in enumerate(crosses)]
            parallels = [parallel if '.exr' in olat_parallel_paths[i][j] else parallel / 255.0 * olat_img_scalar for j, parallel in enumerate(parallels)]
            cross.append(np.einsum('nhwc,nc->hwc', np.stack(crosses, axis=0), olat_omega_i_rgbs[i])) # weighted sum on cross olat images
            parallel.append(np.einsum('nhwc,nc->hwc', np.stack(parallels, axis=0), olat_omega_i_rgbs[i])) # weighted sum on parallel olat images
            nDotL = np.einsum('nc,hwc->nhw', np.stack(olat_omega_i_dirs[i], axis=0), normal_c2w) # (n, h, w)
            irradiance.append(np.einsum('nhw,nc->hwc', np.maximum(nDotL, 0.0), olat_omega_i_rgbs[i])) # (h, w, c), use cross weights as they are usually positive
            if '-random' in self.lighting_augmentation:
                # remove the first one as the first one is usually over exposed
                parallel_stacked.append(np.hstack(parallels[1:]) if len(parallels) > 2 else parallels[0])
            else:
                parallel_stacked.append(np.hstack(parallels) if len(parallels) > 1 else parallels[0]) # for visualization purpose
        cross = np.stack(cross) if len(cross) > 1 else cross[0] # (n, h, w, c)
        parallel = np.stack(parallel) if len(parallel) > 1 else parallel[0]
        irradiance = np.stack(irradiance) if len(irradiance) > 1 else irradiance[0]
        parallel_stacked = np.vstack(parallel_stacked) if len(parallel_stacked) > 1 else parallel_stacked[0] # for visualization purpose
        
        # hdr to ldr via Apply simple Reinhard tone mapping
        # static = self.tonemap.process(static)
        # static = static.clip(0, 1)
        cross = cross.clip(0, 1)
        parallel = parallel.clip(0, 1)
        irradiance = irradiance.clip(0, 1)
        albedo = albedo.clip(0, 1)
        specular = specular.clip(0, 1)
        sigma = sigma.clip(0, 1) / 10. # the decomposition used 10 as a clipping factor
        mask = mask.clip(0, 1)
        
        # [0,1] to [-1,1], as the output will be used for the training of diffusion model
        static = (static - 0.5) * 2.0
        static_cross = (static_cross - 0.5) * 2.0
        static_parallel = (static_parallel - 0.5) * 2.0
        cross = (cross - 0.5) * 2.0
        parallel = (parallel - 0.5) * 2.0
        albedo = (albedo - 0.5) * 2.0
        specular = (specular - 0.5) * 2.0
        sigma = (sigma - 0.5) * 2.0
        irradiance = (irradiance - 0.5) * 2.0 # matching that in the rgb2x pipeline

        # remove nan and inf values
        static = np.nan_to_num(static)
        static_cross = np.nan_to_num(static_cross)
        static_parallel = np.nan_to_num(static_parallel)
        cross = np.nan_to_num(cross)
        parallel = np.nan_to_num(parallel)
        albedo = np.nan_to_num(albedo)
        normal_c2w = np.nan_to_num(normal_c2w)
        normal_w2c = np.nan_to_num(normal_w2c)
        specular = np.nan_to_num(specular)
        sigma = np.nan_to_num(sigma)
        irradiance = np.nan_to_num(irradiance)
        mask = np.nan_to_num(mask)

        # swap x and z to align with the lotus/rgb2x
        # TODO: check rotation matrix as well
        normal_rgb2x_c2w = normal_c2w.copy()
        normal_rgb2x_c2w[:,:,0] *= -1.
        
        # normal is world space normal, transform it to camera space
        normal_rgb2x_w2c = normal_w2c.copy()
        normal_rgb2x_w2c[:,:,0] *= -1.
        
        def hdr_to_ldr(hdr, percentile=95, gamma=2.2, return_int8=False):
            """Convert HDR image to LDR using percentile-based tone mapping."""
            # Compute the percentile value for normalization
            max_val = np.percentile(hdr, percentile)
            # Normalize and clip the values to [0, 1]
            ldr = np.clip(hdr / max_val, 0, 1)
            # Apply gamma correction
            ldr = ldr ** (1.0 / gamma)
            # Scale to [0, 255] and convert to uint8
            if return_int8:
                ldr = (ldr * 255).astype(np.uint8)
            return ldr
        
        if 'exr' in static_path:
            # static is already in [0, 1] range, but we apply tone mapping for consistency
            static = hdr_to_ldr(static)
            static_cross = hdr_to_ldr(static_cross)
            static_parallel = hdr_to_ldr(static_parallel)
        if 'exr' in olat_cross_paths[0]:
            albedo = hdr_to_ldr(albedo)
            

        # apply transforms
        static = self.transforms(static)
        static_cross = self.transforms(static_cross)
        static_parallel = self.transforms(static_parallel)
        cross = torch.tensor(cross).moveaxis(3,1)
        parallel = torch.tensor(parallel).moveaxis(3,1)
        irradiance = torch.tensor(irradiance).moveaxis(3,1)
        albedo = self.transforms(albedo)
        normal_c2w = self.transforms(normal_c2w)
        normal_w2c = self.transforms(normal_w2c)
        normal_rgb2x_c2w = self.transforms(normal_rgb2x_c2w)
        normal_rgb2x_w2c = self.transforms(normal_rgb2x_w2c)
        specular = self.transforms(specular)
        sigma = self.transforms(sigma)
        mask = self.transforms(mask)
        parallel_stacked = self.transforms(parallel_stacked)
        
        # get bounding box
        example['static_value'] = static
        example['static_cross_value'] = static_cross
        example['static_parallel_value'] = static_parallel
        example['cross_value'] = cross
        example['parallel_value'] = parallel
        example['albedo_value'] = albedo
        example['normal_w2c_value'] = normal_w2c
        example['normal_c2w_value'] = normal_c2w # original lightstage normal
        example['normal_rgb2x_w2c_value'] = normal_rgb2x_w2c
        example['normal_rgb2x_c2w_value'] = normal_rgb2x_c2w
        example['specular_value'] = specular.repeat(3, 1, 1) # repeat to 3 channels
        example['sigma_value'] = sigma
        example['irradiance_value'] = irradiance
        example['mask_value'] = mask.bool()
        example['augmented'] = augmented
        example['parallel_value_hstacked'] = parallel_stacked
        
        return example
    
    
def collate_fn_objaverse(examples):
    # meta
    metas = [example['meta'] for example in examples]
    objs = [example['obj_name'] for example in examples]
    # augmented = [example['augmented'] for example in examples]
    obj_materials = [example['obj_material'] for example in examples]
    obj_descriptions = [example['obj_description'] for example in examples]
    camera_paths = [example['camera_path'] for example in examples]
    
    # file paths
    static_paths = [example['static_path'] for example in examples]
    static_cross_paths = [example['static_cross_path'] for example in examples]
    static_parallel_paths = [example['static_parallel_path'] for example in examples]
    cross_paths = [example['cross_path'] for example in examples]
    parallel_paths = [example['parallel_path'] for example in examples]
    olat_omega_i_dirs = [example['olat_omega_i_dirs'] for example in examples]
    olat_omega_i_rgbs = [example['olat_omega_i_rgbs'] for example in examples]
    albedo_paths = [example['albedo_path'] for example in examples]
    normal_paths = [example['normal_path'] for example in examples]
    specular_paths = [example['specular_path'] for example in examples]
    sigma_paths = [example['sigma_path'] for example in examples]
    mask_paths = [example['mask_path'] for example in examples]

    static_values = torch.stack([example['static_value'] for example in examples])
    static_values = static_values.to(memory_format=torch.contiguous_format).float()

    static_cross_values = torch.stack([example['static_cross_value'] for example in examples])
    static_cross_values = static_cross_values.to(memory_format=torch.contiguous_format).float()
    
    static_parallel_values = torch.stack([example['static_parallel_value'] for example in examples])
    static_parallel_values = static_parallel_values.to(memory_format=torch.contiguous_format).float()

    cross_values = torch.stack([example['cross_value'] for example in examples])
    cross_values = cross_values.to(memory_format=torch.contiguous_format).float()

    parallel_values = torch.stack([example['parallel_value'] for example in examples])
    parallel_values = parallel_values.to(memory_format=torch.contiguous_format).float()
    
    parallel_values_hstacked = torch.stack([example['parallel_value_hstacked'] for example in examples])
    parallel_values_hstacked = parallel_values_hstacked.to(memory_format=torch.contiguous_format).float()

    albedo_values = torch.stack([example['albedo_value'] for example in examples])
    albedo_values = albedo_values.to(memory_format=torch.contiguous_format).float()

    normal_ls_w2c_values = torch.stack([example['normal_w2c_value'] for example in examples])
    normal_ls_w2c_values = normal_ls_w2c_values.to(memory_format=torch.contiguous_format).float()

    normal_ls_c2w_values = torch.stack([example['normal_c2w_value'] for example in examples])
    normal_ls_c2w_values = normal_ls_c2w_values.to(memory_format=torch.contiguous_format).float()

    normal_rgb2x_w2c_values = torch.stack([example['normal_rgb2x_w2c_value'] for example in examples])
    normal_rgb2x_w2c_values = normal_rgb2x_w2c_values.to(memory_format=torch.contiguous_format).float()
    
    normal_rgb2x_c2w_values = torch.stack([example['normal_rgb2x_c2w_value'] for example in examples])
    normal_rgb2x_c2w_values = normal_rgb2x_c2w_values.to(memory_format=torch.contiguous_format).float()

    specular_values = torch.stack([example['specular_value'] for example in examples])
    specular_values = specular_values.to(memory_format=torch.contiguous_format).float()

    sigma_values = torch.stack([example['sigma_value'] for example in examples])
    sigma_values = sigma_values.to(memory_format=torch.contiguous_format).float()

    irradiance_values = torch.stack([example['irradiance_value'] for example in examples])
    irradiance_values = irradiance_values.to(memory_format=torch.contiguous_format).float()

    mask_values = torch.stack([example['mask_value'] for example in examples])
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    
    # get pixel values by augment
    pixel_values = torch.stack([example['static_value'] if not example['augmented'] else example['parallel_value'][0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    # get corresponding irradiance values
    pixel_irradiance_values = torch.stack([torch.ones_like(example['irradiance_value'][0]) if not example['augmented'] else example['irradiance_value'][0] for example in examples])
    pixel_irradiance_values = pixel_irradiance_values.to(memory_format=torch.contiguous_format).float()
    
    # augmentation status
    augmented = torch.tensor([example['augmented'] for example in examples], dtype=torch.bool)

    return {
        # values
        # "pixel_values": static_values, # hack
        "pixel_values": pixel_values, # mixed static and parallel values based on augmentation
        "pixel_irradiance_values": pixel_irradiance_values, # irradiance corresponding to the pixel values
        "static_values": static_values,
        "static_cross_values": static_cross_values,
        "static_parallel_values": static_parallel_values,
        "cross_values": cross_values,
        "parallel_values": parallel_values,
        "parallel_values_hstacked": parallel_values_hstacked, # hstacked parallel values for visualization
        "albedo_values": albedo_values,
        "normal_values": normal_rgb2x_w2c_values, # should use rgb2x w2c normal
        "normal_c2w_values": normal_rgb2x_c2w_values,
        "normal_ls_w2c_values": normal_ls_w2c_values, # camera space lightstage
        "normal_ls_c2w_values": normal_ls_c2w_values,
        "specular_values": specular_values,
        # "sigma_values": sigma_values,
        "irradiance_values": irradiance_values,
        "valid_mask_values": mask_values,
        # "noise_values": torch.randn_like(pixel_values), # for diffusion model training
        # paths
        "metas": metas,
        "objs": objs,
        'augmented': augmented, # used for calculating the augmentation ratio in the batch
        'obj_materials': obj_materials,
        'obj_descriptions': obj_descriptions,
        "camera_paths": camera_paths,
        "static_paths": static_paths,
        "static_cross_paths": static_cross_paths,
        "static_parallel_paths": static_parallel_paths,
        "cross_paths": cross_paths,
        "parallel_paths": parallel_paths,
        "olat_omega_i_dirs": olat_omega_i_dirs,
        "olat_omega_i_rgbs": olat_omega_i_rgbs,
        "albedo_paths": albedo_paths,
        "normal_paths": normal_paths,
        "specular_paths": specular_paths,
        "sigma_paths": sigma_paths,
        "mask_paths": mask_paths,
    }