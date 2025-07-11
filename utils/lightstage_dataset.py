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

    def __init__(self, split='train', tasks='', eval_first_n=None):

        assert split in ['train', 'test'], f'Invalid split: {split}'
        
        v = 'v1.3'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_half.json'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_debug.json'
        
        self.root_dir = '/labworking/Users_A-L/jyang/data/LightStageObjectDB'
        self.root_dir = '/home/jyang/data/LightStageObjectDB_test' # local cache, no IO bottle neck
        img_ext = 'exr' # 'exr' or 'jpg' # TODO: jpg need to updated to compatible with negative values, running now
        # meta_data_path = f'{self.root_dir}/datasets/exr/train.json'
        # meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.json'
        # meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.csv'
        meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_fitting_512_ck.csv'
        self.dataset_dir = f'{self.root_dir}/datasets/{img_ext}/{v}/{v}_2'
        self.cam_dir = f'{self.root_dir}/Redline/exr/{v}/{v}_2/cameras'
        
        # load json file
        metadata = []
        with open(meta_data_path) as f:
            if '.json' in meta_data_path:
                metadata = json.load(f)
            elif '.csv' in meta_data_path:
                metadata = pd.read_csv(f).to_dict(orient='records')
        
        self.omega_i_world = self.get_olat()
        # self.bbox_setting = self.init_bbox()
        
        self.texts = []
        self.objs = []
        self.camera_paths = []
        self.static_paths = []
        self.static_cross_paths = []
        self.static_parallel_paths = []
        self.cross_paths = []
        self.parallel_paths = []
        self.albedo_paths = []
        self.normal_paths = []
        self.specular_paths = []
        self.sigma_paths = []
        self.mask_paths = []
        self.omega_i = []
        self.windows = []
        
        print(f"Total files in LightStage dataset at {self.root_dir}: {len(metadata)}")
        # for rowidx, row in enumerate(tqdm(metadata, desc='loading metadata')): # annoying when multi gpu
        for rowidx, row in enumerate(metadata):
            
            # general filter
            if 'fitting' not in meta_data_path:
                if row['l'] <= 1 or row['l'] >= 348:
                    # 2+346+2, 3,695,650 samples
                    continue

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
            train_eval_split = 0.95
            if split == 'train':
                if rowidx / len(metadata) >= train_eval_split:
                    continue
                else:
                    pass
            elif split == 'test':
                if rowidx / len(metadata) < train_eval_split:
                    continue
                else:
                    pass
            
            self.texts.append(row)
            self.objs.append(row["obj"])
            
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
                # cross_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{row["l"]:06d}.{img_ext}')
                # parallel_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{row["l"]:06d}.{img_ext}')
                cross_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_cross.{img_ext}') # hack
                parallel_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'static_parallel.{img_ext}') # hack
                albedo_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'albedo.{img_ext}')
                normal_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'normal.{img_ext}')
                specular_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'specular.{img_ext}')
                sigma_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'sigma.{img_ext}')
                mask_path = os.path.join(self.dataset_dir, f'fit_{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', f'mask.png')
                
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

            self.camera_paths.append(camera_path)
            self.static_paths.append(static_path)
            self.static_cross_paths.append(static_cross_path)
            self.static_parallel_paths.append(static_parallel_path)
            self.cross_paths.append(cross_path)
            self.parallel_paths.append(parallel_path)
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
            eval_idx = list(range(0, len(self.texts), 64))
            eval_idx = eval_idx[:eval_first_n-2]  # -2 for the woodball samples

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
        camera_path = self.camera_paths[idx]
        static_path = self.static_paths[idx]
        cross_path = self.cross_paths[idx]
        parallel_path = self.parallel_paths[idx]
        albedo_path = self.albedo_paths[idx]
        normal_path = self.normal_paths[idx]
        specular_path = self.specular_paths[idx]
        sigma_path = self.sigma_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # check
        example['text'] = self.texts[idx]
        example['obj_name'] = self.objs[idx]
        example['obj_material'] = ''
        example['camera_path'] = self.camera_paths[idx]
        example['static_path'] = self.static_paths[idx]
        example['cross_path'] = self.cross_paths[idx]
        example['parallel_path'] = self.parallel_paths[idx]
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
        cross = imageio.imread(cross_path)
        parallel = imageio.imread(parallel_path)
        albedo = imageio.imread(albedo_path)
        normal = imageio.imread(normal_path)
        specular = imageio.imread(specular_path)
        sigma = imageio.imread(sigma_path)
        mask = imageio.imread(mask_path) if mask_path else (np.ones_like(static[:,:,0], dtype=np.int8) * 255) # mask is optional, use ones if not exist

        # normalize to [0,1]
        static = static if '.exr' in static_path else static / 255.0
        cross = cross if '.exr' in cross_path else cross / 255.0
        parallel = parallel if '.exr' in parallel_path else parallel / 255.0
        albedo = albedo if '.exr' in albedo_path else albedo / 255.0
        normal = normal if '.exr' in normal_path else normal / 255.0
        specular = specular if '.exr' in specular_path else specular / 255.0
        sigma = sigma if '.exr' in sigma_path else sigma / 255.0
        mask = mask / 255.0
        
        # hdr to ldr via Apply simple Reinhard tone mapping
        # static = self.tonemap.process(static)
        static = static.clip(0, 1)
        cross = cross.clip(0, 1)
        parallel = parallel.clip(0, 1)
        albedo = albedo.clip(0, 1)
        normal = normal.clip(-1, 1) if '.exr' in normal_path else (normal.clip(0, 1) * 2. - 1.) # normal is [-1, 1] in exr, [0, 1] in png
        specular = specular.clip(0, 1)
        sigma = sigma.clip(0, 1) / 10. # the decomposition used 10 as a clipping factor
        mask = mask.clip(0, 1)
        
        # remove nan and inf values
        static = np.nan_to_num(static)
        cross = np.nan_to_num(cross)
        parallel = np.nan_to_num(parallel)
        albedo = np.nan_to_num(albedo)
        normal = np.nan_to_num(normal)
        specular = np.nan_to_num(specular)
        sigma = np.nan_to_num(sigma)
        mask = np.nan_to_num(mask)
        
        # normal is world space normal, transform it to camera space
        normal_w2c = np.einsum('ij, hwi -> hwj', R, normal)

        # swap x and z to align with the lotus/rgb2x
        normal = normal[:, :, [2, 1, 0]]
        normal_w2c = normal_w2c[:, :, [2, 1, 0]]

        # apply transforms
        static = self.transforms(static)
        cross = self.transforms(cross)
        parallel = self.transforms(parallel)
        albedo = self.transforms(albedo)
        normal = self.transforms(normal)
        normal_w2c = self.transforms(normal_w2c)
        specular = self.transforms(specular)
        sigma = self.transforms(sigma)
        mask = self.transforms(mask)
        
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

    return {
        # values
        # "static_values": static_values,
        "pixel_values": static_values, # hack
        "cross_values": cross_values,
        "parallel_values": parallel_values,
        # "albedo_values": albedo_values,
        "diffuse_values": albedo_values,
        "normal_values": normal_w2c_values, # camera space normal
        "normal_c2w_values": normal_c2w_values,
        "specular_values": specular_values,
        "sigma_values": sigma_values,
        "valid_mask_values": mask_values,
        # paths
        "static_pathes": static_pathes,
        "cross_pathes": cross_pathes,
        "parallel_pathes": parallel_pathes,
        # "albedo_pathes": albedo_pathes,
        "diffuse_pathes": albedo_pathes,
        "normal_pathes": normal_pathes,
        "specular_pathes": specular_pathes,
        "sigma_pathes": sigma_pathes,
        "mask_pathes": mask_pathes,
    }