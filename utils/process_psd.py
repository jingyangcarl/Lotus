import os
import cv2
import numpy as np
from tqdm.rich import tqdm

if __name__ == "__main__":
    
    root_in = '/labworking/Users/jyang/data/psd/PSD_Dataset/'
    root_out = '/labworking/Users/jyang/data/psd/PSD_Dataset_processed/'
    
    def get_specular(rgb_image, rgb_diffuse):
        rgb_specular = rgb_image.astype(np.int16) - rgb_diffuse.astype(np.int16)
        rgb_specular = np.clip(rgb_specular, 0, 255).astype(np.uint8)
        return rgb_specular
    
    splits = ['train', 'val', 'test']
    for split in splits:
        
        # for paired data
        # images_dir = os.path.join(root_in, f'PSD_{split.capitalize()}', f'PSD_{split.capitalize()}_specular') # image with diffuse+specular
        # images_path = list(os.scandir(images_dir))
        # for image_path in tqdm(images_path, desc=f"Processing {split} dataset"):
            
        #     file_name = image_path.name
        #     src_image_path = image_path.path
        #     src_diffuse_path = image_path.path.replace('_specular', '_diffuse')
            
        #     if not os.path.exists(src_diffuse_path) or not os.path.exists(src_image_path):
        #         print(f"File not found: {src_image_path} or {src_diffuse_path}")
        #         continue
            
        #     # specular is not available and need to be calculated
        #     rgb_image = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
        #     rgb_diffuse = cv2.imread(src_diffuse_path, cv2.IMREAD_UNCHANGED)
        #     rgb_specular = get_specular(rgb_image, rgb_diffuse)
            
        #     # save image, diffuse and specular
        #     dst_image_path = os.path.join(root_out, split, f'image', file_name)
        #     dst_diffuse_path = os.path.join(root_out, split, f'diffuse', file_name)
        #     dst_specular_path = os.path.join(root_out, split, f'specular', file_name)
            
        #     os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        #     os.makedirs(os.path.dirname(dst_diffuse_path), exist_ok=True)
        #     os.makedirs(os.path.dirname(dst_specular_path), exist_ok=True)
            
        #     cv2.imwrite(dst_image_path, rgb_image)
        #     cv2.imwrite(dst_diffuse_path, rgb_diffuse)
        #     cv2.imwrite(dst_specular_path, rgb_specular)
    
        # # for grouped data
        # subfolders = ['aligned', 'unaligned'] if split == 'train' else ['']
        # for subfolder in subfolders:
        #     images_dir = os.path.join(root_in, f'PSD_{split.capitalize()}', f'PSD_{split.capitalize()}_group', subfolder) # image with diffuse+specular
        #     images_path = list(os.scandir(images_dir))
        #     images_path = ['-'.join(p.path.split('-')[:-1]) for p in images_path]
            
        #     # remove duplicates and keep orders
        #     images_path = list(dict.fromkeys(images_path))
            
        #     for image_path in tqdm(images_path, desc=f'processing {split} group data'):
                
        #         file_name = os.path.basename(image_path)
                
        #         for i in range(2,13):
        #             src_image_path = image_path + f'-{i:02d}.png'
        #             src_diffuse_path = image_path + f'-{1:02d}.png'
        #             file_name = os.path.basename(src_image_path)
                    
        #             if not os.path.exists(src_diffuse_path) or not os.path.exists(src_image_path):
        #                 print(f"File not found: {src_image_path} or {src_diffuse_path}")
        #                 continue
                    
        #             # specular is not available and need to be calculated
        #             rgb_image = cv2.imread(src_image_path, cv2.IMREAD_UNCHANGED)
        #             rgb_diffuse = cv2.imread(src_diffuse_path, cv2.IMREAD_UNCHANGED)
        #             rgb_specular = get_specular(rgb_image, rgb_diffuse)
                    
        #             # save image, diffuse and specular
        #             dst_image_path = os.path.join(root_out, split, f'image', file_name)
        #             dst_diffuse_path = os.path.join(root_out, split, f'diffuse', file_name)
        #             dst_specular_path = os.path.join(root_out, split, f'specular', file_name)
                    
        #             os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        #             os.makedirs(os.path.dirname(dst_diffuse_path), exist_ok=True)
        #             os.makedirs(os.path.dirname(dst_specular_path), exist_ok=True)
                    
        #             cv2.imwrite(dst_image_path, rgb_image)
        #             cv2.imwrite(dst_diffuse_path, rgb_diffuse)
        #             cv2.imwrite(dst_specular_path, rgb_specular)           
                    
        # loop through the images and output as an mp4 vide
        def create_video_from_images(root_out, split, frame_size=(256, 256), fps=30):
            # Define paths
            images_dir = os.path.join(root_out, split, 'image')
            diffuse_dir = os.path.join(root_out, split, 'diffuse')
            specular_dir = os.path.join(root_out, split, 'specular')
            video_path = os.path.join(root_out, split, 'image.mp4')

            # Get sorted list of image paths
            images_path = sorted([entry.path for entry in os.scandir(images_dir) if entry.is_file()])

            # Define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

            for image_path in tqdm(images_path, desc=f"Creating video for {split} dataset"):
                image = cv2.imread(image_path)
                diffuse = cv2.imread(image_path.replace('image', 'diffuse'))
                specular = cv2.imread(image_path.replace('image', 'specular'))
                if image is None:
                    print(f"Warning: Could not read {image_path}, skipping.")
                    continue
                image_resized = cv2.resize(image, frame_size)
                diffuse_resized = cv2.resize(diffuse, frame_size)
                specular_resized = cv2.resize(specular, frame_size)
                frame = np.hstack((image_resized, diffuse_resized, specular_resized))
                out.write(frame)

            out.release()
            print(f"Video saved at {video_path}")
            
        create_video_from_images(root_out, split)
            