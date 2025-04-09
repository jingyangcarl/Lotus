import shutil
import os
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import cv2
from tqdm import tqdm

def copy_realworld_and_run(db_src, db_dst):
    
    imgs = os.listdir(db_src)
    for img in imgs:
        
        if 'md' in img:
            continue
        
        img_name = img.split('.')[0]
        src = os.path.join(db_src, img)
        dst = os.path.join(db_dst, img_name, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # print(f'Copying {src} to {dst}')
        # shutil.copy(src, dst)
        
        # instead of copy, we read and write via donwsample
        img = imageio.imread(src)
        if img.shape[0] > 512:
            down_size = img.shape[0] // 512
            img = cv2.resize(img, (img.shape[1] // down_size, img.shape[0] // down_size))
        
        # makesure img size is interger multiple of 32
        img = img[:img.shape[0] // 32 * 32, :img.shape[1] // 32 * 32, :]
        imageio.imwrite(dst, img)
        
        # run rgbx pipeline
        test_folder = os.path.join(db_dst, img_name)
        test_folder_relative = os.path.relpath(test_folder, db_dst_root)
        try:
            subprocess.run([
                'python', 'infer.py', 
                '--pretrained_model_name_or_path=jingheya/lotus-normal-g-v1-0',
                '--prediction_type="sample"',
                '--seed=42',
                '--half_precision',
                f'--input_dir={os.path.dirname(dst)}',
                '--task_name=normal',
                '--mode=generation',
                f'--output_dir={test_folder}',
                '--disparity'
            ])
            
        except Exception as e:
            print(f'Error running iid pipeline for: {e}')
            continue
            


def render_olats(rets, camera, cam_downscale=8.0, resize_shape=None):
    
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
            
        if resize_shape is not None:
            H, W = resize_shape
            focal = focal * np.array([W, H]) / resolution
            pp = pp * np.array([W, H]) / resolution
            resolution = np.array([W, H])
            
                
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
    
    cam_path = '/home/jyang/projects/ObjectReal/data/LightStageObjectDB/Redline/v1.2/v1.2_2/cameras'
    camid = camera[-2:]
    cam = get_lightstage_camera(os.path.join(cam_path, f'camera{camid}.txt'), downscale=cam_downscale)
    
    # build view_dirs
    H, W = cam['hwf'][:2]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cam['pp'][0])/cam['K'][0,0], -(j-cam['pp'][1])/cam['K'][1,1], -np.ones_like(i)], -1)
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6)

    # build lighting dirs
    olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
    olat_pos_ = np.genfromtxt(f'{olat_base}/LSX3_light_positions.txt').astype(np.float32)
    olat_idx = np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
    r = R.from_euler('y', 180, degrees=True)
    olat_pos_ = (olat_pos_ @ r.as_matrix().T).astype(np.float32)
    omega_i_world = olat_pos_[olat_idx-1]
    
    # get material properties, Image to numpy
    albedo = np.asarray(rets[0][0]) / 255.0
    normal = np.asarray(rets[1][0]) / 255.0 * 2.0 - 1.0
    roughness = np.asarray(rets[2][0]) / 255.0
    metallic = np.asarray(rets[3][0]) / 255.0
    irradiance = np.asarray(rets[4][0]) / 255.0
    
    # ensure normal vectors are normalized
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
    
    # Define reflectance function (Cook-Torrance BRDF)
    def fresnel_schlick(cos_theta, F0):
        """Schlick's approximation for Fresnel term"""
        return F0 + (1 - F0) * np.power(1 - cos_theta, 5)

    def ggx_distribution(NdotH, alpha):
        """GGX normal distribution function (NDF)"""
        alpha2 = alpha * alpha
        denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0) ** 2
        return alpha2 / (np.pi * denom)

    def smith_schlick_ggx(NdotV, NdotL, alpha):
        """Smith Schlick-GGX Geometry function"""
        k = (alpha + 1) ** 2 / 8.0
        G1 = NdotV / (NdotV * (1 - k) + k)
        G2 = NdotL / (NdotL * (1 - k) + k)
        return G1 * G2

    def cook_torrance_brdf(N, V, L, albedo, roughness, metallic):
        """Cook-Torrance BRDF computation"""
        H = (V + L) / np.linalg.norm(V + L, axis=-1, keepdims=True)
        
        NdotL = np.maximum(np.sum(N * L, axis=-1, keepdims=True), 1e-6)
        NdotV = np.maximum(np.sum(N * V, axis=-1, keepdims=True), 1e-6)
        NdotH = np.maximum(np.sum(N * H, axis=-1, keepdims=True), 1e-6)
        VdotH = np.maximum(np.sum(V * H, axis=-1, keepdims=True), 1e-6)
        
        # F0 for metals and dielectrics
        F0 = 0.04 * (1 - metallic) + albedo * metallic
        F = fresnel_schlick(VdotH, F0)
        
        D = ggx_distribution(NdotH, roughness ** 2)
        G = smith_schlick_ggx(NdotV, NdotL, roughness ** 2)
        
        denominator = 4 * NdotV * NdotL + 1e-6
        specular = (D * F * G) / denominator
        
        k_s = F
        k_d = (1 - k_s) * (1 - metallic)
        
        diffuse = (albedo / np.pi) * k_d
        
        return (diffuse + specular) * NdotL, NdotL

    # Compute shading
    V = -dirs  # View direction (negate view vectors)
    
    rendered_olat = []
    rendered_ndotl = []
    for i in tqdm(range(omega_i_world.shape[0])):
        L = omega_i_world[i]  # Light direction
        L = L / np.linalg.norm(L)  # Normalize
        
        # Compute Cook-Torrance shading
        shading, ndotl = cook_torrance_brdf(normal, V, L, albedo, roughness, metallic)
        
        # Apply irradiance
        exposure = 100
        shaded_image = shading * irradiance * exposure
        
        rendered_olat.append(shaded_image)
        rendered_ndotl.append(ndotl)
    
    return rendered_olat, rendered_ndotl
            

def copy_LightStageObjectDB_and_run(db_src, db_dst, db_dst_root, obj_list):
    
    objects = os.listdir(db_src)
    for obj in objects:
        
        if obj == 'cameras':
            continue # skip cameras folder
        
        if obj_list and obj not in obj_list:
            continue # skip objects not in obj_list
        
        pattern = 'static'
        
        cameras = os.listdir(os.path.join(db_src, obj, pattern))
        for camera in cameras:
            
            # copy mixed_w2.jpg
            src = os.path.join(db_src, obj, pattern, camera, 'mixed_w2.jpg')
            dst = os.path.join(db_dst, obj, pattern, camera, 'mixed_w2.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # print(f'Copying {src} to {dst}')
            shutil.copy(src, dst)
            
            img = imageio.imread(src)
            
            # run iid pipeline
            test_folder = os.path.join(db_dst, obj, pattern, camera)
            test_folder_relative = os.path.relpath(test_folder, db_dst_root)
            try:
                
                subprocess.run([
                    'python', 'infer.py', 
                    '--pretrained_model_name_or_path=jingheya/lotus-normal-g-v1-0',
                    '--prediction_type="sample"',
                    '--seed=42',
                    '--half_precision',
                    f'--input_dir={os.path.dirname(dst)}',
                    '--task_name=normal',
                    '--mode=generation',
                    f'--output_dir={test_folder}',
                    '--disparity'
                ])
                
            except Exception as e:
                print(f'Error running iid pipeline for {test_folder_relative}: {e}')
                continue
            
            # exit() # run later tonight

if __name__ == '__main__':
    
    v, res = 'v1.2', 8
    db_src_root = '/labworking/Users_A-L/jyang/data'
    db_dst_root = '/home/jyang/projects/Lotus/data'
    db_src = db_src_root + f'/LightStageObjectDB/Redline/jpg/{v}/{v}_{res}'
    db_dst = db_dst_root + f'/LightStageObjectDB/Redline/jpg/{v}/{v}_{res}'
    
    obj_list = []

    # disable the openexternal when running the code
    # Press Ctrl + Shift + P and type: Preferences: Open Settings (JSON)
    # Add the following entry: "window.openExternal": false
    
    # copy_LightStageObjectDB_and_run(db_src, db_dst, db_dst_root, obj_list)
    
    db_src = '/home/jyang/projects/ObjectReal/data/realworld'
    db_dst = '/home/jyang/projects/Lotus/data/realworld'
    copy_realworld_and_run(db_src, db_dst)
    
    
    
    
    