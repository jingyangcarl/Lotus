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

    def __init__(self):
        
        v = 'v1.3'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_half.json'
        # metadata_path = f'./data/matnet/train/matnet_olat_{v}_debug.json'
        
        self.root_dir = '/labworking/Users_A-L/jyang/data/LightStageObjectDB'
        img_ext = 'exr' # 'exr' or 'jpg'
        # meta_data_path = f'{self.root_dir}/datasets/exr/train.json'
        # meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.json'
        meta_data_path = f'{self.root_dir}/datasets/exr/{v}/{v}_2/train_512_.csv'
        self.dataset_dir = f'{self.root_dir}/datasets/{img_ext}/{v}/{v}_2'
        self.img_dir = f'{self.root_dir}/Processed/{v}/{v}_2'
        self.olat_dir = f'{self.root_dir}/Redline/{img_ext}/{v}/{v}_2'
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
        self.omega_i = []
        self.windows = []
        
        print(f"Total files in LightStage dataset at {self.root_dir}: {len(metadata)}")
        for _, row in enumerate(tqdm(metadata[:1000], desc='loading metadata')):
            
            if row['l'] <= 1 or row['l'] >= 348:
                # 2+346+2, 3,695,650 samples
                continue
            
            if row['l'] != 2:
                continue # verify the diffuse specular removal, 10559 samples
            
            self.texts.append(row["obj"])
            self.objs.append(row["obj"])
            
            camera_path = os.path.join(self.cam_dir, f'camera{row["cam"]:02d}.txt')
            static_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static', f'{row["i"]}_{row["j"]}.{img_ext}')
            static_cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_cross', f'{row["i"]}_{row["j"]}.{img_ext}')
            static_parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'static_parallel', f'{row["i"]}_{row["j"]}.{img_ext}')
            cross_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'cross', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{img_ext}')
            parallel_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'parallel', f'{row["i"]}_{row["j"]}.{row["l"]:06d}.{img_ext}')
            albedo_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'albedo', f'{row["i"]}_{row["j"]}.{img_ext}')
            normal_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'normal', f'{row["i"]}_{row["j"]}.{img_ext}')
            specular_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'specular', f'{row["i"]}_{row["j"]}.{img_ext}')
            sigma_path = os.path.join(self.dataset_dir, f'{row["res"]}', row["obj"], f'cam{row["cam"]:02d}', 'sigma', f'{row["i"]}_{row["j"]}.{img_ext}')
            
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
            
            self.omega_i.append(self.omega_i_world[row['l']-2]) # 2+346+2
            self.windows.append((row['i'], row['j'], row['res']))
            
        
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
    
    # def init_bbox(self):
    #     settings = {
    #         # v1.2
    #         # 'name', [cam00, cam01, cam02, cam03, cam04, cam05, cam06, cam07]
    #         # 'yucake': {'HW': (1900, 1350), 'LT': [(0, 200), (250, 250), (250, 50), (100, 180), (250, 150), (250, 320), (200, 470), (400, 250)]},
    #         # '8020': {'HW': (2000, 600), 'LT': [(800, 50), (1000, 120), (700, 0), (400, 0), (450, 50), (300, 200), (500, 0), (650, 100)]},
    #         # 'eiffeltower': {'HW': (1350, 800), 'LT': [(550, 700), (800, 800), (500, 300), (250, 380), (400, 700), (350, 850), (400, 300), (500, 800)]},
    #         # 'elephant': {'HW': (1350, 1080), 'LT': [(400, 700), (540, 750), (400, 300), (200, 380), (300, 700), (250, 800), (300, 300), (400, 750)]},
    #         # 'speccup': {'HW': (1150, 700), 'LT': [(750, 900), (900, 1000), (600, 550), (350, 600), (400, 950), (250, 1050), (500, 600), (600, 1000)]},
    #         # 'sodacan': {'HW': (1080, 600), 'LT': [(750, 1000), (940, 1100), (600, 700), (400, 850), (500, 1050), (350, 1200), (550, 700), (650, 1100)]},
    #         # 'mug': {'HW': (1080, 1080), 'LT': [(350, 1000), (540, 1100), (400, 700), (200, 850), (300, 1050), (300, 1200), (300, 700), (400, 1150)]},
    #         # 'specapple': {'HW': (1080, 1080), 'LT': [(450, 1050), (540, 1100), (400, 700), (150, 800), (250, 1000), (150, 1150), (300, 800), (450, 1100)]},
    #         # 'pcb': {'HW': (900, 700), 'LT': [(600, 1150), (850, 1250), (550, 900), (300, 1000), (450, 1150), (350, 1300), (450, 900), (600, 1300)]},
    #         # 'glove': {'HW': (1300, 800), 'LT': [(600, 750), (820, 850), (600, 500), (350, 580), (400, 780), (250, 900), (550, 400), (650, 830)]},
    #         # 'toy': {'HW': (800, 700), 'LT': [(600, 1250), (850, 1350), (550, 1000), (300, 1100), (450, 1250), (350, 1400), (450, 1000), (600, 1350)]},
    #         # 'woodcube': {'HW': (900, 900), 'LT': [(700, 1200), (720, 1300), (500, 800), (150, 950), (220, 1250), (100, 1400), (400, 700), (500, 1350)]},
    #         # 'paintedcube': {'HW': (900, 900), 'LT': [(720, 1200), (720, 1300), (500, 800), (150, 950), (220, 1250), (70, 1400), (400, 700), (500, 1350)]},
    #         # 'fabricbag': {'HW': (1080, 1080), 'LT': [(450, 1000), (540, 1100), (400, 700), (150, 850), (250, 1050), (150, 1200), (300, 700), (400, 1100)]},
    #         # 'mirrorball': {'HW': (500, 500), 'LT': [(700, 1450), (950, 1550), (680, 1200), (450, 1300), (600, 1450), (530, 1600), (600, 1300), (730, 1550)]},
    #         # 'greyball': {'HW': (500, 500), 'LT': [(700, 1450), (950, 1550), (680, 1200), (450, 1300), (600, 1450), (530, 1600), (600, 1300), (730, 1550)]},
    #         # 'manikin': {'HW': (2100, 1550), 'LT': [(100, 0), (240, 150), (400, 300), (200, 380), (300, 700), (250, 800), (300, 300), (400, 250)]},
    #         # v1.3
    #         ## session: 20240508_ObjectDB
    #         # 'pillowwhite': {'HW': (2500, 1600), 'LT': [(0, 250), (0, 400), (0, 0), (0, 200), (0, 200), (10, 1300), (0, 200), (0, 300)]}, # 5 failed
    #         'recyclepaperbox': {'HW': (2200, 1500), 'LT': [(0, 480), (100, 600), (100, 200), (0, 350), (50, 500), (100, 600), (0, 400), (50, 550)]},                # check: pattern (static X after mixed_w2 on cam4), cropping O
    #         'osbbox': {'HW': (1800, 1100), 'LT': [(550, 1000), (500, 1100), (350, 600), (150, 750), (100, 1000), (0, 1050), (200, 650), (250, 1050)]},              # check: pattern O, cropping O
    #         'woodenbox': {'HW': (1800, 900), 'LT': [(550, 1000), (550, 1100), (400, 600), (200, 750), (200, 1000), (100, 1100), (300, 650), (350, 1050)]},          # check: pattern O, cropping O
    #         'remakesheet1': {'HW': (2200, 1200), 'LT': [(400, 480), (400, 600), (300, 150), (100, 300), (100, 500), (0, 700), (200, 200), (150, 550)]},             # check: pattern O, cropping O
    #         'remakesheet2': {'HW': (2200, 1200), 'LT': [(400, 480), (400, 600), (300, 150), (100, 300), (100, 500), (0, 700), (200, 200), (150, 550)]},             # check: pattern (static X after mixed_w2 on cam6), cropping O
    #         'remakesheet3': {'HW': (2200, 1200), 'LT': [(400, 480), (400, 600), (300, 150), (100, 300), (100, 500), (0, 700), (200, 200), (150, 550)]},             # check: pattern O, cropping O
    #         'remakesheet4': {'HW': (2200, 1200), 'LT': [(400, 480), (400, 600), (300, 150), (100, 300), (100, 500), (0, 700), (200, 200), (150, 550)]},             # check: pattern O, cropping O
    #         'plasticgrass': {'HW': (2200, 1000), 'LT': [(450, 480), (450, 600), (350, 150), (180, 300), (200, 500), (100, 700), (250, 200), (300, 550)]},           # check: pattern O, cropping O
    #         'pan': {'HW': (1200, 1200), 'LT': [(500, 1600), (500, 1700), (400, 1400), (200, 1500), (200, 1650), (100, 1750), (300, 1400), (350, 1700)]},            # check: pattern O, cropping O
    #         'book1pink': {'HW': (1800, 1400), 'LT': [(0, 880), (0, 1000), (0, 550), (200, 700), (300, 900), (150, 1050), (100, 600), (150, 1000)]},                 # check: pattern O, cropping O
    #         'book2orange': {'HW': (1800, 1400), 'LT': [(150, 900), (50, 1000), (0, 550), (100, 700), (300, 900), (150, 1050), (100, 600), (50, 1000)]},             # check: pattern O, cropping O
    #         'book3yellow': {'HW': (1800, 1400), 'LT': [(150, 900), (50, 1000), (0, 550), (100, 700), (300, 900), (150, 1050), (100, 600), (150, 1000)]},            # check: pattern O, cropping O
    #         'book4green': {'HW': (1800, 1400), 'LT': [(200, 900), (50, 1000), (50, 550), (100, 700), (200, 900), (0, 1050), (100, 600), (150, 1000)]},              # check: pattern O, cropping O
    #         'book5blue': {'HW': (1800, 1400), 'LT': [(250, 900), (200, 1000), (50, 550), (100, 700), (150, 900), (0, 1050), (100, 600), (150, 1000)]},              # check: pattern O, cropping O
    #         'brick': {'HW': (1500, 800), 'LT': [(500, 1200), (450, 1300), (400, 850), (250, 1000), (300, 1200), (250, 1350), (300, 1000), (350, 1300)]},            # check: pattern O, cropping O
    #         'pillowblack': {'HW': (1600, 1600), 'LT': [(0, 1450), (0, 1450), (0, 900), (0, 1100), (0, 1450), (10, 1450), (0, 700), (0, 1600)]},                     # check: pattern O, cropping O
    #         ## session: 20240506_ObjectDB
    #         'handbaglogo1white': {'HW': (1600, 1600), 'LT': [(0, 1150), (0, 1270), (0, 800), (0, 900), (0, 1200), (10, 1300), (0, 800), (0, 1300)]},                # check: pattern O, cropping O
    #         'handbaglogo2brown': {'HW': (1500, 1500), 'LT': [(200, 1250), (200, 1370), (100, 900), (0, 1050), (0, 1250), (0, 1400), (0, 800), (0, 1350)]},          # check: pattern (static X after mixed_w2 on cam5), cropping O
    #         'handbaglogo3black': {'HW': (1500, 1500), 'LT': [(200, 950), (200, 1070), (100, 600), (0, 750), (0, 950), (10, 1000), (0, 700), (0, 1050)]},            # check: pattern O, cropping O
    #         'handbaggrid1white': {'HW': (1400, 1000), 'LT': [(500, 1070), (500, 1170), (350, 750), (100, 950), (150, 1100), (100, 1200), (250, 900), (300, 1150)]}, # check: pattern (static X after mixed_w2 on cam6,7), cropping O
    #         'handbaggrid2black': {'HW': (1400, 1000), 'LT': [(400, 1070), (400, 1170), (330, 780), (200, 980), (250, 1100), (200, 1200), (250, 900), (300, 1180)]}, # check: pattern O, cropping O
    #         'lundarynet1white': {'HW': (2300, 1500), 'LT': [(500, 300), (500, 400), (300, 100), (0, 250), (50, 400), (0, 500), (100, 200), (100, 400)]},            # check: pattern O, cropping O
    #         'lundarynet2black': {'HW': (2300, 1500), 'LT': [(300, 300), (100, 400), (100, 100), (0, 250), (50, 400), (0, 500), (0, 400), (100, 400)]},              # check: pattern (static X after mixed_w2 on cam6), cropping O
    #         'pouchbagfabric1blue': {'HW': (1500, 1500), 'LT': [(100, 1000), (100, 1100), (50, 700), (0, 850), (0, 1000), (0, 1100), (0, 800), (0, 1000)]},          # check: pattern O, cropping O
    #         'pouchbagfabric2black': {'HW': (1500, 1500), 'LT': [(100, 1000), (100, 1100), (50, 700), (0, 850), (0, 1000), (0, 1100), (0, 800), (0, 1000)]},         # check: pattern O, cropping O
    #         'furbag1white': {'HW': (1500, 1200), 'LT': [(300, 850), (350, 930), (300, 600), (150, 750), (150, 850), (50, 1000), (200, 700), (250, 900)]},           # check: pattern O, cropping O
    #         'furbag2brown': {'HW': (1500, 1200), 'LT': [(350, 800), (350, 900), (300, 500), (100, 700), (100, 850), (0, 950), (150, 700), (200, 900)]},             # check: pattern (static X after mixed_w2 on cam2,6), cropping O
    #         'silkbag1green': {'HW': (1400, 1100), 'LT': [(400, 900), (400, 1000), (300, 600), (150, 800), (150, 950), (0, 1050), (230, 800), (280, 1000)]},         # check: pattern O, cropping O
    #         'silkbag2blue': {'HW': (1400, 1100), 'LT': [(400, 900), (400, 1000), (300, 600), (150, 800), (150, 950), (0, 1050), (230, 800), (280, 1000)]},          # check: pattern O, cropping O
    #         'silkbag3pink': {'HW': (1400, 1100), 'LT': [(400, 900), (400, 1000), (300, 600), (150, 800), (150, 950), (0, 1050), (230, 800), (280, 1000)]},          # check: pattern O, cropping O
    #         'silkbag4purple': {'HW': (1400, 1100), 'LT': [(400, 900), (400, 1000), (300, 600), (150, 800), (150, 950), (0, 1050), (230, 800), (280, 1000)]},        # check: pattern O, cropping O
    #         'silkbag5yellow': {'HW': (1400, 1100), 'LT': [(400, 900), (400, 1000), (300, 600), (150, 800), (150, 950), (0, 1050), (230, 800), (280, 1000)]},        # check: pattern O, cropping O
    #         'bolsabag': {'HW': (1400, 1300), 'LT': [(300, 900), (300, 1000), (300, 600), (50, 800), (50, 950), (0, 1050), (100, 800), (180, 1000)]},                # check: pattern O, cropping O
    #         'bottleholder': {'HW': (1400, 600), 'LT': [(650, 900), (650, 1000), (550, 600), (400, 800), (400, 950), (300, 1050), (450, 800), (500, 1000)]},         # check: pattern O, cropping O
    #         'showerball1mint': {'HW': (1000, 1000), 'LT': [(450, 900), (450, 1000), (400, 600), (200, 800), (200, 950), (110, 1050), (300, 800), (350, 1000)]},     # check: pattern O, cropping O
    #         'showerball2purple': {'HW': (1000, 1000), 'LT': [(500, 950), (450, 1050), (400, 700), (200, 850), (200, 950), (110, 1100), (300, 800), (350, 1050)]},   # check: pattern O, cropping O
    #         'showerball3blue': {'HW': (1000, 1000), 'LT': [(500, 950), (450, 1100), (400, 700), (200, 850), (200, 1000), (110, 1150), (300, 800), (350, 1100)]},    # check: pattern O, cropping O
    #         'showerball4pink': {'HW': (1000, 1000), 'LT': [(500, 950), (500, 1100), (400, 700), (200, 850), (200, 1000), (110, 1150), (300, 800), (350, 1050)]},    # check: pattern O, cropping O
    #         'woodrabbitAblue': {'HW': (1200, 700), 'LT': [(600, 450), (600, 550), (550, 200), (350, 400), (400, 500), (300, 600), (450, 500), (500, 500)]},         # check: pattern O, cropping O
    #         'woodrabbitBpink': {'HW': (900, 1000), 'LT': [(500, 720), (450, 800), (400, 500), (220, 650), (250, 750), (150, 850), (300, 700), (350, 800)]},         # check: pattern O, cropping O
    #         'woodrabbitBblue': {'HW': (900, 1000), 'LT': [(500, 720), (450, 800), (400, 500), (220, 650), (250, 750), (150, 850), (300, 700), (350, 800)]},         # check: pattern O, cropping O
    #         'woodrabbitCblue': {'HW': (1000, 1000), 'LT': [(500, 550), (450, 650), (450, 300), (320, 450), (300, 550), (150, 650), (400, 500), (450, 600)]},        # check: pattern O, cropping O
    #         'woodrabbitCpink': {'HW': (1000, 1000), 'LT': [(500, 550), (450, 650), (450, 300), (320, 450), (300, 550), (150, 650), (400, 500), (450, 600)]},        # check: pattern O, cropping O
    #         'plasticbucketred': {'HW': (1000, 1000), 'LT': [(450, 650), (450, 750), (400, 350), (220, 550), (200, 700), (150, 850), (300, 500), (350, 750)]},       # check: pattern O, cropping O
    #         'metalbucketwhite': {'HW': (900, 900), 'LT': [(500, 800), (480, 900), (400, 500), (220, 650), (200, 800), (150, 950), (300, 630), (350, 850)]},         # check: pattern O, cropping O
    #         'metalbottlegold': {'HW': (1100, 800), 'LT': [(500, 600), (500, 700), (450, 300), (300, 450), (300, 600), (200, 750), (350, 500), (400, 650)]},         # check: pattern O, cropping O
    #         'plasticboxwhite': {'HW': (1200, 700), 'LT': [(600, 400), (550, 500), (550, 100), (300, 300), (300, 400), (200, 550), (400, 400), (450, 450)]},         # check: pattern O, cropping O
    #         'sponge1red': {'HW': (800, 600), 'LT': [(650, 800), (620, 900), (550, 520), (350, 700), (350, 820), (300, 950), (450, 720), (490, 880)]},               # check: pattern O, cropping O
    #         'sponge2orange': {'HW': (800, 600), 'LT': [(650, 800), (650, 900), (580, 520), (380, 700), (380, 820), (300, 950), (470, 720), (520, 880)]},            # check: pattern O, cropping O
    #         'sponge3yellow': {'HW': (800, 600), 'LT': [(650, 800), (650, 900), (580, 520), (380, 700), (380, 820), (300, 950), (470, 720), (520, 880)]},            # check: pattern O, cropping O
    #         'sponge4green': {'HW': (800, 600), 'LT': [(650, 800), (650, 900), (580, 520), (380, 700), (380, 820), (300, 950), (470, 720), (520, 880)]},             # check: pattern O, cropping O
    #         'sponge5blue': {'HW': (800, 600), 'LT': [(650, 800), (650, 900), (580, 520), (380, 700), (380, 820), (300, 950), (470, 720), (520, 880)]},              # check: pattern O, cropping O
    #         ## session: 20240426_ObjectDB
    #         'manikincolor': {'HW': (2200, 1500), 'LT': [(100, 850), (100, 870), (50, 600), (0, 750), (0, 850), (10, 950), (0, 700), (0, 920)]},     # check: pattern (static X after mixed_w2 on cam3, 4, 7), cropping O
    #         'cushioncover1': {'HW': (2300, 1600), 'LT': [(0, 750), (0, 770), (0, 700), (0, 700), (0, 700), (10, 700), (0, 700), (0, 700)]},         # check: pattern O, cropping O
    #         'cushioncover2': {'HW': (2300, 1600), 'LT': [(0, 750), (0, 770), (0, 700), (0, 700), (0, 700), (10, 700), (0, 700), (0, 700)]},         # check: pattern O, cropping O
    #         'cushioncover3': {'HW': (2300, 1600), 'LT': [(0, 750), (0, 770), (0, 700), (0, 700), (0, 700), (10, 700), (0, 700), (0, 700)]},         # check: pattern O, cropping O
    #         'cushioncover4': {'HW': (2300, 1600), 'LT': [(0, 750), (0, 770), (0, 700), (0, 700), (0, 700), (10, 700), (0, 700), (0, 700)]},         # check: pattern O, cropping O
    #         'tablecloth1': {'HW': (2200, 1600), 'LT': [(0, 850), (0, 870), (0, 650), (0, 800), (0, 800), (10, 800), (0, 800), (0, 800)]},           # check: pattern O, cropping O
    #         'tablecloth2': {'HW': (2200, 1600), 'LT': [(0, 850), (0, 870), (0, 650), (0, 800), (0, 800), (10, 800), (0, 800), (0, 800)]},           # check: pattern O, cropping O
    #         'tablecloth3': {'HW': (2200, 1600), 'LT': [(0, 850), (0, 870), (0, 650), (0, 800), (0, 800), (10, 800), (0, 800), (0, 800)]},           # check: pattern (static X after mixed_w2 on cam2), cropping O
    #         'tablecloth4': {'HW': (2200, 1600), 'LT': [(0, 850), (0, 870), (0, 650), (0, 800), (0, 800), (10, 800), (0, 800), (0, 800)]},           # check: pattern O, cropping O
    #         'tablecloth5': {'HW': (2200, 1600), 'LT': [(0, 850), (0, 870), (0, 650), (0, 800), (0, 800), (10, 800), (0, 800), (0, 800)]},           # check: pattern O, cropping O
    #         'petmat': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 500), (0, 800)]},                # check: pattern O, cropping O
    #         'slipperhanging1': {'HW': (2200, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},       # check: pattern O, cropping O
    #         'curtain': {'HW': (2200, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},               # check: pattern (static X after mixed_w2 on cam1), cropping O
    #         'totebag1': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},              # check: pattern O, cropping O
    #         'totebag2': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},              # check: pattern O, cropping O
    #         'aluminiumbag': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 900)]},          # check: pattern O, cropping O
    #         'reusablebag1': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},          # check: pattern O, cropping O
    #         'reusablebag2': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},          # check: pattern O, cropping O
    #         'reusablebag3': {'HW': (1800, 1600), 'LT': [(0, 750), (0, 870), (0, 400), (0, 500), (0, 700), (10, 900), (0, 400), (0, 800)]},          # check: pattern O, cropping O
    #         ## session: 20240425_ObjectDB
    #         'carrot': {'HW': (2000, 400), 'LT': [(550, 550), (550, 650), (550, 400), (400, 500), (550, 550), (500, 650), (500, 500), (550, 650)]},                              # check: pattern O, cropping O
    #         'purpleonion': {'HW': (800, 700), 'LT': [(400, 1950), (400, 2050), (400, 1650), (300, 1800), (400, 1950), (400, 2050), (350, 1800), (400, 2050)]},                  # check: pattern O, cropping O
    #         'whiteonion': {'HW': (600, 900), 'LT': [(300, 2200), (350, 2300), (300, 1900), (200, 2050), (300, 2150), (300, 2250), (250, 2000), (300, 2300)]},                   # check: pattern O, cropping O
    #         'eggplant': {'HW': (1300, 700), 'LT': [(400, 1500), (450, 1600), (450, 1250), (350, 1400), (450, 1500), (400, 1600), (350, 1400), (400, 1600)]},                    # check: pattern O, cropping O
    #         'cucumber': {'HW': (1500, 500), 'LT': [(550, 1150), (550, 1250), (500, 900), (400, 1100), (500, 1150), (500, 1250), (450, 1000), (500, 1250)]},                     # check: pattern O, cropping O
    #         'greencabbage': {'HW': (1000, 1000), 'LT': [(500, 1680), (480, 1780), (400, 1350), (200, 1500), (220, 1680), (120, 1800), (300, 1400), (350, 1800)]},               # check: pattern O, cropping O
    #         'purplecabbage': {'HW': (1000, 1000), 'LT': [(500, 1780), (480, 1880), (400, 1450), (200, 1600), (220, 1780), (120, 1900), (300, 1400), (350, 1900)]},              # check: pattern O, cropping O
    #         'broccoli': {'HW': (1400, 1100), 'LT': [(300, 1200), (250, 1300), (150, 900), (0, 1100), (50, 1150), (50, 1300), (50, 1000), (50, 1250)]},                          # check: pattern O, cropping O
    #         'potato': {'HW': (700, 600), 'LT': [(400, 2050), (400, 2180), (400, 1800), (300, 1950), (500, 2050), (500, 2150), (350, 1950), (400, 2180)]},                       # check: pattern O, cropping O
    #         'sweetpotato': {'HW': (1200, 700), 'LT': [(300, 1350), (300, 1500), (250, 1150), (200, 1300), (350, 1350), (400, 1450), (200, 1400), (250, 1450)]},                 # check: pattern O, cropping O
    #         'redapple': {'HW': (600, 600), 'LT': [(480, 2150), (480, 2280), (430, 1850), (300, 2050), (420, 2150), (400, 2250), (350, 2000), (400, 2280)]},                     # check: pattern O, cropping O
    #         'greenapple': {'HW': (600, 600), 'LT': [(450, 2200), (450, 2300), (400, 1900), (300, 2050), (450, 2200), (430, 2300), (350, 2050), (400, 2300)]},                   # check: pattern O, cropping O
    #         'pear': {'HW': (700, 600), 'LT': [(450, 2100), (450, 2200), (400, 1800), (300, 1950), (450, 2100), (430, 2200), (350, 1950), (400, 2200)]},                         # check: pattern O, cropping O
    #         'redbellpepper': {'HW': (900, 800), 'LT': [(350, 1850), (400, 1950), (400, 1550), (200, 1700), (300, 1800), (300, 1950), (250, 1650), (300, 1950)]},                # check: pattern O, cropping O
    #         'orangebellpepper': {'HW': (800, 800), 'LT': [(350, 2050), (400, 2150), (350, 1750), (200, 1900), (400, 2000), (350, 2150), (300, 1900), (300, 2150)]},             # check: pattern O, cropping O
    #         'yellowbellpepper': {'HW': (900, 800), 'LT': [(350, 1850), (400, 1950), (350, 1550), (250, 1700), (350, 1850), (350, 1950), (300, 1650), (350, 1950)]},             # check: pattern O, cropping O
    #         'corn': {'HW': (2000, 600), 'LT': [(450, 650), (450, 750), (400, 300), (300, 500), (400, 650), (400, 750), (300, 500), (350, 700)]},                                # check: pattern O, cropping O
    #         'cornpeel': {'HW': (2000, 1200), 'LT': [(50, 1050), (50, 1050), (100, 900), (0, 1050), (200, 1050), (200, 1050), (100, 1050), (100, 1050)]},                        # check: pattern O, cropping O
    #         'greenonion': {'HW': (2000, 700), 'LT': [(150, 700), (300, 800), (300, 500), (400, 650), (600, 650), (600, 750), (400, 850), (400, 750)]},                          # check: pattern O, cropping O
    #         'celery': {'HW': (2500, 800), 'LT': [(350, 100), (500, 200), (450, 0), (350, 150), (500, 150), (450, 250), (400, 250), (450, 250)]},                                # check: pattern O, cropping O
    #         'choucavalier': {'HW': (2100, 1200), 'LT': [(300, 600), (350, 650), (300, 300), (200, 450), (300, 550), (150, 650), (250, 450), (300, 600)]},                       # check: pattern O, cropping O
    #         'banana': {'HW': (1100, 1500), 'LT': [(0, 1680), (50, 1780), (50, 1350), (0, 1500), (120, 1650), (120, 1700), (0, 1500), (50, 1700)]},                              # check: pattern O, cropping O
    #         'papaya': {'HW': (1000, 1500), 'LT': [(0, 1780), (50, 1880), (50, 1450), (0, 1600), (120, 1750), (120, 1800), (0, 1600), (50, 1800)]},                              # check: pattern O, cropping O
    #         'orangereal': {'HW': (700, 700), 'LT': [(350, 2050), (400, 2200), (400, 1800), (300, 1950), (450, 2050), (450, 2150), (320, 1950), (400, 2200)]},                   # check: pattern O, cropping O
    #         'mango': {'HW': (800, 700), 'LT': [(400, 1950), (400, 2050), (400, 1650), (250, 1850), (400, 1950), (400, 2050), (300, 1800), (400, 2050)]},                        # check: pattern O, cropping O
    #         'kiwano': {'HW': (900, 700), 'LT': [(400, 1850), (400, 2000), (400, 1550), (250, 1750), (400, 1850), (400, 1950), (300, 1700), (400, 2000)]},                       # check: pattern O, cropping O
    #         'dragondruit': {'HW': (1000, 800), 'LT': [(400, 1750), (400, 1870), (350, 1450), (200, 1650), (350, 1750), (300, 1850), (250, 1550), (300, 1880)]},                 # check: pattern O, cropping O
    #         'avacado': {'HW': (600, 500), 'LT': [(500, 2130), (500, 2240), (450, 1870), (350, 2030), (500, 2130), (500, 2230), (420, 2000), (480, 2250)]},                      # check: pattern O, cropping O
    #         'watermelon': {'HW': (1300, 1300), 'LT': [(0, 1400), (50, 1500), (80, 1150), (50, 1300), (220, 1350), (250, 1500), (50, 1300), (100, 1500)]},                       # check: pattern O, cropping O
    #         'cantaloupe': {'HW': (1200, 1200), 'LT': [(100, 1500), (150, 1600), (180, 1250), (150, 1400), (320, 1450), (350, 1600), (150, 1400), (200, 1600)]},                 # check: pattern O, cropping O
    #         'honeydew': {'HW': (1200, 1200), 'LT': [(100, 1500), (150, 1600), (180, 1250), (150, 1400), (320, 1450), (350, 1600), (150, 1400), (200, 1600)]},                   # check: pattern O, cropping O
    #         'pineapple': {'HW': (2500, 1100), 'LT': [(300, 200), (300, 300), (250, 0), (100, 0), (200, 200), (150, 300), (200, 0), (250, 250)]},                                # check: pattern O, cropping O
    #         ## session: 20240403_ObjectDB
    #         'whitethermocolball1': {'HW': (700, 700), 'LT': [(300, 1450), (400, 1560), (450, 1250), (450, 1400), (580, 1450), (570, 1580), (430, 1480), (480, 1560)]},          # check: pattern O, cropping O
    #         'redchristmasball1': {'HW': (700, 700), 'LT': [(300, 1500), (400, 1600), (450, 1300), (450, 1450), (580, 1500), (570, 1620), (430, 1530), (480, 1590)]},            # check: pattern O, cropping O
    #         'redchristmasball2': {'HW': (700, 700), 'LT': [(300, 1500), (400, 1600), (450, 1300), (450, 1450), (580, 1500), (570, 1620), (430, 1530), (480, 1590)]},            # check: pattern O, cropping O
    #         'crystalball': {'HW': (600, 600), 'LT': [(350, 1600), (450, 1700), (500, 1400), (500, 1550), (620, 1600), (620, 1700), (480, 1600), (530, 1680)]},                  # check: pattern O, cropping O
    #         'greychristmasball1': {'HW': (600, 600), 'LT': [(350, 1600), (450, 1700), (500, 1400), (500, 1550), (620, 1600), (620, 1700), (480, 1600), (530, 1680)]},           # check: pattern O, cropping O
    #         'greychristmasball2': {'HW': (600, 600), 'LT': [(350, 1580), (450, 1680), (500, 1380), (500, 1550), (620, 1580), (620, 1700), (480, 1600), (530, 1680)]},           # check: pattern O, cropping O
    #         'whitethermocolball2': {'HW': (600, 600), 'LT': [(350, 1580), (450, 1680), (500, 1380), (500, 1550), (620, 1580), (620, 1700), (480, 1600), (530, 1680)]},          # check: pattern O, cropping O
    #         'whitechristmasballshiny': {'HW': (600, 600), 'LT': [(350, 1580), (400, 1680), (400, 1350), (350, 1500), (520, 1580), (520, 1700), (350, 1550), (400, 1680)]},      # check: pattern O, cropping O
    #         'redchristmasballshiny': {'HW': (500, 500), 'LT': [(400, 1680), (450, 1780), (450, 1430), (400, 1600), (550, 1650), (550, 1800), (400, 1630), (450, 1750)]},        # check: pattern O, cropping O
    #         'whitechristmasballshiny2': {'HW': (700, 700), 'LT': [(300, 1500), (350, 1600), (350, 1250), (300, 1450), (450, 1500), (450, 1620), (300, 1450), (350, 1590)]},     # check: pattern O, cropping O
    #         'rubberwhiteball': {'HW': (600, 600), 'LT': [(350, 1580), (400, 1680), (400, 1350), (350, 1500), (520, 1580), (520, 1700), (350, 1550), (400, 1680)]},              # check: pattern O, cropping O
    #         'rubberredball': {'HW': (500, 500), 'LT': [(400, 1680), (450, 1780), (450, 1430), (400, 1600), (550, 1650), (550, 1800), (400, 1630), (450, 1750)]},                # check: pattern O, cropping O
    #         'blackballclearcoat': {'HW': (600, 600), 'LT': [(350, 1580), (400, 1680), (400, 1350), (350, 1500), (520, 1580), (520, 1700), (350, 1550), (400, 1680)]},           # check: pattern O, cropping O
    #         'redchristmasballblur': {'HW': (500, 500), 'LT': [(400, 1680), (450, 1780), (450, 1430), (400, 1600), (550, 1650), (550, 1800), (400, 1630), (450, 1750)]},         # check: pattern O, cropping O
    #         'woodball': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1700), (600, 1780), (600, 1900), (450, 1730), (500, 1880)]},                     # check: pattern O, cropping O
    #         'metalballrotten': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1700), (600, 1780), (600, 1900), (450, 1730), (500, 1880)]},              # check: pattern O, cropping O
    #         'candleball': {'HW': (600, 600), 'LT': [(350, 1580), (400, 1680), (400, 1350), (350, 1500), (520, 1580), (520, 1700), (350, 1550), (400, 1680)]},                   # check: pattern O, cropping O
    #         'whitethermocolball3': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1700), (600, 1780), (600, 1900), (450, 1730), (500, 1880)]},          # check: pattern O, cropping O
    #         'plasticgreyball': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1700), (600, 1780), (600, 1900), (450, 1730), (500, 1880)]},              # check: pattern O, cropping O
    #         'marbalegg': {'HW': (500, 400), 'LT': [(450, 1680), (500, 1780), (500, 1430), (450, 1600), (600, 1680), (600, 1800), (450, 1630), (500, 1780)]},                    # check: pattern O, cropping O
    #         'marbalball': {'HW': (500, 500), 'LT': [(400, 1680), (450, 1780), (450, 1430), (400, 1600), (550, 1650), (550, 1800), (400, 1630), (450, 1750)]},                   # check: pattern O, cropping O
    #         'greenchristmasball': {'HW': (500, 500), 'LT': [(400, 1680), (450, 1780), (450, 1430), (400, 1600), (550, 1680), (550, 1800), (400, 1630), (450, 1800)]},           # check: pattern O, cropping O
    #         'greenchristmasballshiny': {'HW': (500, 500), 'LT': [(400, 1650), (450, 1760), (450, 1430), (400, 1600), (550, 1650), (550, 1780), (400, 1630), (450, 1760)]},      # check: pattern O, cropping O
    #         'anisotropyball': {'HW': (300, 300), 'LT': [(500, 1870), (540, 1980), (550, 1630), (500, 1800), (650, 1850), (650, 1980), (500, 1820), (560, 1970)]},               # check: pattern O, cropping O
    #         'rubberblueball': {'HW': (700, 700), 'LT': [(300, 1460), (350, 1570), (350, 1250), (300, 1400), (450, 1450), (450, 1580), (300, 1450), (350, 1560)]},               # check: pattern O, cropping O
    #         'redplasticball': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1720), (600, 1780), (600, 1900), (450, 1730), (500, 1900)]},               # check: pattern O, cropping O
    #         'greenplasticball': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1720), (600, 1780), (600, 1900), (450, 1730), (500, 1900)]},             # check: pattern O, cropping O
    #         'blueplasticball': {'HW': (400, 400), 'LT': [(450, 1780), (500, 1880), (500, 1530), (450, 1720), (600, 1780), (600, 1900), (450, 1730), (500, 1900)]},              # check: pattern O, cropping O
    #         ## session: 20240327_ObjectDB
    #         'whitehead': {'HW': (2200, 1500), 'LT': [(100, 500), (100, 600), (50, 300), (0, 500), (50, 550), (10, 650), (0, 400), (100, 600)]},                     # check: pattern O, cropping O
    #         'woodenclock': {'HW': (1800, 1500), 'LT': [(0, 900), (0, 1000), (50, 700), (0, 900), (100, 950), (100, 1050), (0, 900), (100, 1050)]},                  # check: pattern O, cropping O
    #         'jarcandy': {'HW': (1650, 800), 'LT': [(300, 1050), (400, 1150), (400, 800), (400, 1000), (500, 1050), (450, 1200), (350, 1000), (400, 1150)]},         # check: pattern O, cropping O
    #         'candlebottle': {'HW': (1800, 800), 'LT': [(300, 950), (400, 1050), (400, 700), (400, 900), (500, 950), (450, 1100), (350, 900), (400, 1050)]},         # check: pattern O, cropping O
    #         'candle': {'HW': (1000, 700), 'LT': [(350, 1680), (450, 1780), (450, 1400), (450, 1550), (550, 1680), (500, 1800), (450, 1400), (500, 1790)]},          # check: pattern O, cropping O
    #         'ceramicsbottle': {'HW': (1000, 700), 'LT': [(350, 1700), (450, 1800), (450, 1400), (450, 1600), (550, 1700), (500, 1800), (450, 1400), (500, 1800)]},  # check: pattern O, cropping O
    #         'ceramicstoy': {'HW': (800, 1000), 'LT': [(350, 1900), (350, 2000), (350, 1600), (300, 1800), (350, 1900), (350, 2000), (300, 1700), (350, 2000)]},     # check: pattern O, cropping O
    #         'ceramicsbumer': {'HW': (800, 800), 'LT': [(400, 1850), (400, 2000), (450, 1550), (400, 1750), (450, 1900), (400, 2000), (400, 1600), (450, 2000)]},    # check: pattern O, cropping O
    #         'ceramicsjar': {'HW': (800, 700), 'LT': [(400, 1850), (450, 2000), (450, 1550), (400, 1750), (500, 1870), (500, 2000), (400, 1600), (450, 1980)]},      # check: pattern O, cropping O
    #         'ceramicscup': {'HW': (900, 700), 'LT': [(400, 1800), (450, 1900), (450, 1450), (400, 1650), (500, 1770), (500, 1900), (400, 1500), (450, 1900)]},      # check: pattern O, cropping O
    #         'plastictoy': {'HW': (900, 600), 'LT': [(500, 1800), (600, 1900), (600, 1450), (500, 1650), (550, 1770), (500, 1900), (500, 1600), (550, 1900)]},       # check: pattern O, cropping O
    #         'cleybear': {'HW': (900, 1000), 'LT': [(350, 1800), (450, 1950), (400, 1450), (300, 1650), (300, 1800), (200, 1950), (300, 1600), (400, 1950)]},        # check: pattern O, cropping O
    #         'bleydolphin': {'HW': (900, 1200), 'LT': [(250, 1850), (150, 1950), (150, 1450), (150, 1650), (250, 1800), (200, 1950), (150, 1600), (150, 1950)]},     # check: pattern O, cropping O
    #         'glassjar': {'HW': (1100, 800), 'LT': [(300, 1600), (400, 1700), (400, 1300), (350, 1500), (450, 1600), (400, 1700), (350, 1400), (400, 1700)]},        # check: pattern O, cropping O
    #         'glassbottle': {'HW': (800, 700), 'LT': [(400, 1900), (450, 2000), (450, 1550), (400, 1750), (500, 1870), (500, 2000), (400, 1600), (450, 2000)]},      # check: pattern O, cropping O
    #         'transparentbar': {'HW': (1000, 550), 'LT': [(400, 1700), (450, 1800), (500, 1450), (500, 1550), (600, 1670), (600, 1800), (450, 1500), (530, 1800)]},  # check: pattern O, cropping O
    #         'plastichead': {'HW': (800, 1100), 'LT': [(0, 1950), (0, 2050), (100, 1700), (250, 1900), (450, 1950), (450, 2050), (100, 1900), (200, 2100)]},         # check: pattern O, cropping O
    #         'fabricbagblue': {'HW': (900, 900), 'LT': [(250, 1800), (350, 1900), (450, 1500), (500, 1650), (550, 1800), (500, 1950), (450, 1600), (450, 1900)]},    # check: pattern O, cropping O
    #         'fabricbagred': {'HW': (1000, 1000), 'LT': [(250, 1650), (250, 1750), (250, 1400), (300, 1550), (450, 1650), (400, 1800), (300, 1400), (300, 1800)]},   # check: pattern O, cropping O
    #         'fabricbox': {'HW': (1100, 800), 'LT': [(250, 1600), (350, 1700), (400, 1300), (350, 1450), (450, 1600), (500, 1700), (350, 1400), (400, 1700)]},       # check: pattern O, cropping O
    #         ## session: 20240318_ObjectDB
    #         'toybear': {'HW': (2200, 1500), 'LT': [(0, 800), (100, 900), (50, 600), (0, 800), (50, 850), (10, 850), (0, 700), (0, 900)]},                           # check: pattern O, cropping O
    #         'toykoala': {'HW': (2500, 1600), 'LT': [(0, 300), (20, 500), (20, 200), (0, 400), (20, 450), (10, 500), (0, 400), (0, 400)]},                           # check: pattern O, cropping O
    #         'toypenguin': {'HW': (2200, 1500), 'LT': [(0, 800), (100, 900), (50, 600), (0, 800), (50, 850), (10, 850), (0, 700), (0, 900)]},                        # check: pattern O, cropping O
    #         'toydear': {'HW': (1600, 1050), 'LT': [(150, 1100), (150, 1250), (150, 900), (200, 1000), (350, 1100), (350, 1250), (100, 1000), (200, 1200)]},         # check: pattern O, cropping O
    #         'toydragon': {'HW': (2000, 1500), 'LT': [(0, 800), (100, 900), (50, 600), (0, 800), (50, 850), (10, 950), (0, 700), (0, 900)]},                         # check: pattern O, cropping O
    #         'toymonster': {'HW': (2900, 1600), 'LT': [(0, 160), (0, 160), (0, 160), (0, 160), (0, 160), (0, 160), (0, 160), (0, 160)]},                             # check: pattern O, cropping O
    #         'toydolphin': {'HW': (1100, 900), 'LT': [(350, 1600), (250, 1700), (300, 1200), (300, 1400), (350, 1600), (250, 1700), (350, 1200), (400, 1700)]},      # check: pattern O, cropping O
    #         'plastermask': {'HW': (1200, 1200), 'LT': [(0, 1300), (0, 1400), (100, 1200), (400, 1400), (400, 1400), (400, 1450), (200, 1600), (300, 1550)]},        # check: pattern (static X after mixed_w2 on cam7), cropping O
    #         'plasterhead': {'HW': (1300, 1500), 'LT': [(0, 1400), (0, 1500), (0, 1300), (100, 1400), (100, 1400), (100, 1500), (0, 1500), (0, 1500)]},              # check: pattern O, cropping O
    #         'ceramicspot': {'HW': (1300, 1400), 'LT': [(0, 1400), (0, 1500), (0, 1100), (100, 1300), (200, 1400), (200, 1500), (0, 1200), (50, 1500)]},             # check: pattern O, cropping O
    #         'humidifier': {'HW': (1500, 1100), 'LT': [(100, 1200), (200, 1300), (200, 900), (200, 1100), (350, 1200), (300, 1300), (200, 1000), (250, 1350)]},      # check: pattern O, cropping O
    #         'paperballon': {'HW': (550, 550), 'LT': [(450, 2150), (500, 2250), (500, 1850), (450, 2050), (500, 2150), (500, 2250), (450, 1950), (500, 2250)]},      # check: pattern O, cropping O
    #         'paperbox': {'HW': (1000, 1000), 'LT': [(50, 1750), (250, 1900), (250, 1500), (300, 1700), (400, 1750), (450, 1900), (250, 1500), (300, 1900)]},        # check: pattern O, cropping O
    #         'bamboosteamer': {'HW': (1200, 1200), 'LT': [(50, 1550), (150, 1650), (200, 1250), (200, 1400), (300, 1550), (300, 1650), (150, 1400), (250, 1650)]},   # check: pattern O, cropping O
    #         'coffeecan': {'HW': (1100, 800), 'LT': [(250, 1600), (300, 1700), (350, 1300), (400, 1500), (500, 1600), (500, 1700), (350, 1400), (400, 1700)]},       # check: pattern O, cropping O
    #         'woodenholder': {'HW': (1900, 800), 'LT': [(250, 800), (300, 900), (400, 500), (400, 700), (450, 800), (500, 900), (350, 700), (400, 900)]},            # check: pattern O, cropping O
    #         'concrete1': {'HW': (1200, 1500), 'LT': [(0, 1550), (0, 1650), (0, 1350), (0, 1500), (100, 1600), (100, 1750), (0, 1600), (0, 1750)]},                  # check: pattern O, cropping O
    #         'concrete2': {'HW': (1000, 1000), 'LT': [(350, 1750), (350, 1850), (250, 1400), (200, 1600), (350, 1750), (350, 1900), (150, 1500), (250, 1850)]},      # check: pattern O, cropping O
    #         'concrete3': {'HW': (1100, 1100), 'LT': [(100, 1600), (200, 1700), (200, 1300), (300, 1500), (350, 1600), (400, 1750), (300, 1500), (350, 1750)]},      # check: pattern O, cropping O
    #         'paperbag': {'HW': (2000, 1600), 'LT': [(0, 700), (100, 800), (50, 500), (0, 700), (10, 750), (10, 850), (0, 900), (0, 800)]},                          # check: pattern O, cropping O
    #         ## session: 20240311_ObjectDB
    #         'blackplastic': {'HW': (900, 650), 'LT': [(650, 1400), (600, 1500), (550, 1100), (350, 1200), (350, 1400), (350, 1550), (450, 1200), (500, 1500)]},     # check: pattern O, cropping O
    #         'metalkettle': {'HW': (1500, 1500), 'LT': [(0, 1100), (100, 1200), (100, 800), (100, 1000), (100, 1000), (100, 1150), (100, 800), (100, 1200)],},       # check: pattern O, cropping O
    #         'orange': {'HW': (700, 700), 'LT': [(660, 1550), (700, 1650), (650, 1200), (500, 1400), (450, 1600), (350, 1700), (570, 1300), (600, 1680)]},           # check: pattern O, cropping O
    #         'clorox': {'HW': (1700, 800), 'LT': [(550, 600), (500, 700), (500, 300), (400, 500), (350, 650), (300, 800), (400, 400), (450, 700)]},                  # check: pattern O, cropping O
    #         'owlcase': {'HW': (900, 900), 'LT': [(400, 1350), (420, 1500), (400, 1050), (250, 1250), (300, 1400), (250, 1500), (300, 1100), (350, 1470)]},          # check: pattern O, cropping O
    #         'metalbear': {'HW': (1350, 800), 'LT': [(550, 1000), (540, 1100), (500, 700), (350, 800), (300, 1000), (250, 1100), (400, 700), (400, 1050)]},          # check: pattern O, cropping O
    #         'pottedplant1': {'HW': (1000, 1100), 'LT': [(400, 1350), (400, 1400), (350, 1000), (250, 1150), (250, 1400), (150, 1500), (300, 1000), (350, 1450)]},   # check: pattern O, cropping O
    #         'pinecone': {'HW': (1100, 900), 'LT': [(450, 1200), (500, 1300), (450, 950), (300, 1050), (350, 1200), (250, 1350), (350, 1000), (400, 1300)]},         # check: pattern O, cropping O
    #         'tumbler': {'HW': (1100, 900), 'LT': [(450, 1200), (500, 1300), (450, 950), (300, 1050), (350, 1200), (250, 1350), (350, 1000), (400, 1300)]},          # check: pattern O, cropping O
    #         'woodencat': {'HW': (1700, 800), 'LT': [(500, 600), (500, 700), (500, 300), (400, 500), (350, 650), (300, 800), (400, 400), (450, 700)]},               # check: pattern O, cropping O
    #         'flowervase': {'HW': (1300, 1100), 'LT': [(400, 1000), (400, 1100), (350, 700), (250, 900), (250, 1050), (100, 1150), (280, 750), (350, 1100)]},        # check: pattern (static X after mixed_w2 on cam6), cropping O
    #         'greypurse': {'HW': (2100, 1350), 'LT': [(400, 200), (250, 250), (250, 0), (100, 100), (150, 200), (0, 350), (200, 0), (300, 270)]},                    # check: pattern O, cropping O
    #         'plant1': {'HW': (900, 1100), 'LT': [(300, 1450), (400, 1500), (400, 1100), (350, 1300), (300, 1500), (250, 1600), (400, 1200), (400, 1600)]},          # check: pattern O, cropping O
    #         'pottedplant2': {'HW': (1200, 1100), 'LT': [(500, 1100), (500, 1200), (450, 750), (300, 900), (150, 1100), (50, 1250), (350, 750), (400, 1200)]},       # check: pattern O, cropping O
    #         'football': {'HW': (1300, 1300), 'LT': [(200, 1000), (250, 1100), (350, 750), (300, 900), (300, 1000), (250, 1150), (250, 750), (400, 1100)]},          # check: pattern (static X after mixed_w2 on cam5), cropping O
    #         'coachbag': {'HW': (2300, 1500), 'LT': [(50, 50), (100, 250), (150, 0), (100, 0), (50, 100), (0, 250), (200, 0), (300, 150)]},                          # check: pattern O, cropping O
    #         'stonepot': {'HW': (1000, 1000), 'LT': [(350, 1350), (400, 1450), (350, 1000), (250, 1150), (300, 1350), (200, 1500), (300, 1000), (350, 1450)]},       # check: pattern O, cropping O
    #         'metalbox': {'HW': (1400, 1000), 'LT': [(450, 900), (500, 1000), (500, 600), (300, 750), (300, 950), (200, 1050), (400, 700), (450, 1000)]},            # check: pattern O, cropping O
    #         'branches': {'HW': (2100, 1350), 'LT': [(400, 200), (250, 250), (250, 0), (100, 100), (150, 200), (0, 350), (200, 0), (200, 270)]},                     # check: pattern (static X after mixed_w2 on cam5,7), cropping O
    #         'sodacan': {'HW': (1080, 600), 'LT': [(750, 1200), (740, 1300), (700, 900), (500, 1000), (500, 1250), (350, 1350), (600, 900), (650, 1300)]},           # check: pattern O, cropping O
    #         'mirrorball': {'HW': (500, 500), 'LT': [(550, 1250), (600, 1350), (550, 1050), (500, 1200), (550, 1250), (500, 1400), (500, 1250), (550, 1350)]},       # check: pattern O, cropping O (statis redline image are missmatched)
    #         'greyball': {'HW': (500, 500), 'LT': [(550, 1250), (600, 1350), (550, 1050), (500, 1200), (550, 1250), (500, 1400), (500, 1250), (550, 1350)]},         # check: pattern O, cropping O
    #     }
        
    #     return settings
    
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
        
        # check
        example['text'] = self.texts[idx]
        example['camera_path'] = self.camera_paths[idx]
        example['static_path'] = self.static_paths[idx]
        example['cross_path'] = self.cross_paths[idx]
        example['parallel_path'] = self.parallel_paths[idx]
        example['albedo_path'] = self.albedo_paths[idx]
        example['normal_path'] = self.normal_paths[idx]
        example['specular_path'] = self.specular_paths[idx]
        example['sigma_path'] = self.sigma_paths[idx]
        
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
        
        # hdr to ldr via Apply simple Reinhard tone mapping
        # static = self.tonemap.process(static)
        static = static.clip(0, 1)
        cross = cross.clip(0, 1)
        parallel = parallel.clip(0, 1)
        albedo = albedo.clip(0, 1)
        normal = normal.clip(0, 1)
        specular = specular.clip(0, 1)
        sigma = sigma.clip(0, 1) / 10. # the decomposition used 10 as a clipping factor
        
        # remove nan and inf values
        static = np.nan_to_num(static)
        cross = np.nan_to_num(cross)
        parallel = np.nan_to_num(parallel)
        albedo = np.nan_to_num(albedo)
        normal = np.nan_to_num(normal)
        specular = np.nan_to_num(specular)
        sigma = np.nan_to_num(sigma)
        
        # apply transforms
        static = self.transforms(static)
        cross = self.transforms(cross)
        parallel = self.transforms(parallel)
        albedo = self.transforms(albedo)
        normal = self.transforms(normal)
        specular = self.transforms(specular)
        sigma = self.transforms(sigma)
        
        # get bounding box
        example['static_value'] = static
        example['cross_value'] = cross
        example['parallel_value'] = parallel
        example['albedo_value'] = albedo
        example['normal_value'] = normal
        example['specular_value'] = specular.repeat(3, 1, 1) # repeat to 3 channels
        example['sigma_value'] = sigma
        
        return example
    
    
def collate_fn_lightstage(examples):
    static_pathes = [example['static_path'] for example in examples]
    cross_pathes = [example['cross_path'] for example in examples]
    parallel_pathes = [example['parallel_path'] for example in examples]
    albedo_pathes = [example['albedo_path'] for example in examples]
    normal_pathes = [example['normal_path'] for example in examples]
    specular_pathes = [example['specular_path'] for example in examples]
    sigma_pathes = [example['sigma_path'] for example in examples]

    static_values = torch.stack([example['static_value'] for example in examples])
    static_values = static_values.to(memory_format=torch.contiguous_format).float()

    cross_values = torch.stack([example['cross_value'] for example in examples])
    cross_values = cross_values.to(memory_format=torch.contiguous_format).float()

    parallel_values = torch.stack([example['parallel_value'] for example in examples])
    parallel_values = parallel_values.to(memory_format=torch.contiguous_format).float()

    albedo_values = torch.stack([example['albedo_value'] for example in examples])
    albedo_values = albedo_values.to(memory_format=torch.contiguous_format).float()

    normal_values = torch.stack([example['normal_value'] for example in examples])
    normal_values = normal_values.to(memory_format=torch.contiguous_format).float()

    specular_values = torch.stack([example['specular_value'] for example in examples])
    specular_values = specular_values.to(memory_format=torch.contiguous_format).float()

    sigma_values = torch.stack([example['sigma_value'] for example in examples])
    sigma_values = sigma_values.to(memory_format=torch.contiguous_format).float()

    return {
        # values
        # "static_values": static_values,
        "pixel_values": static_values, # hack
        "cross_values": cross_values,
        "parallel_values": parallel_values,
        # "albedo_values": albedo_values,
        "diffuse_values": albedo_values,
        "normal_values": normal_values,
        "specular_values": specular_values,
        "sigma_values": sigma_values,
        # paths
        "static_pathes": static_pathes,
        "cross_pathes": cross_pathes,
        "parallel_pathes": parallel_pathes,
        # "albedo_pathes": albedo_pathes,
        "diffuse_pathes": albedo_pathes,
        "normal_pathes": normal_pathes,
        "specular_pathes": specular_pathes,
        "sigma_pathes": sigma_pathes,
    }