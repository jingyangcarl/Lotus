import os
from datasets import Dataset as Dataset_hf
from tqdm.rich import tqdm
from PIL import Image
from torchvision import transforms
import torch

def get_PSD_dataset(data_dir, split='train'):
    
    split_dir = os.path.join(data_dir, split)
    
    # load data and construct the dataset
    data_dict = {
        "image": [], # 
        "diffuse": [],
        "specular": [],
    }
    file_count = sum(len(files) for _, _, files in os.walk(split_dir))  # Get the number of files
    print(f"Total files in PSD dataset at {split_dir}: {file_count}")
    with tqdm(total=file_count, desc=f"Walking through SHIQ dataset") as pbar:
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                pbar.update(1)
                image_path = os.path.join(root, file)
                diffuse_path = image_path.replace("image/", "diffuse/")
                specular_path = image_path.replace("image/", "specular/")
                
                # the process is secured by process_psd, there fore no need to check existence
                
                data_dict["image"].append(image_path)
                data_dict["diffuse"].append(diffuse_path)
                data_dict["specular"].append(specular_path)
    dataset = Dataset_hf.from_dict(data_dict)
    
    # define dataset transform
    column_names = dataset.column_names

    image_column = column_names[0]
    diffuse_column = column_names[1]
    specular_colum = column_names[2]
    
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
        # transforms.Resize((256, 256)),
    ])
    
    
    def preprocess_psd(examples):
        # convert image to RGB
        images = [Image.open(image).convert("RGB") for image in examples[image_column]]
        diffuse = [Image.open(diffuse).convert("RGB") for diffuse in examples[diffuse_column]]
        specular = [Image.open(specular).convert("RGB") for specular in examples[specular_colum]]
        
        examples["pixel_values"] = []
        examples["diffuse_values"] = []
        examples["specular_values"] = []
        
        for image, diffuse, specular in zip(images, diffuse, specular):
            
            train_image = transform(image)
            train_diffuse = transform(diffuse)
            train_specular = transform(specular)
            
            examples["pixel_values"].append(train_image)
            examples["diffuse_values"].append(train_diffuse)
            examples["specular_values"].append(train_specular)
            
        return examples
    
    
    def collate_fn_psd(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        diffuse_values = torch.stack([example["diffuse_values"] for example in examples])
        diffuse_values = diffuse_values.to(memory_format=torch.contiguous_format).float()
        
        specular_values = torch.stack([example["specular_values"] for example in examples])
        specular_values = specular_values.to(memory_format=torch.contiguous_format).float()
        
        image_paths = [example[image_column] for example in examples]
        diffuse_paths = [example[diffuse_column] for example in examples]
        specular_paths = [example[specular_colum] for example in examples]
        
        example_dict = {
            "pixel_values": pixel_values,
            "diffuse_values": diffuse_values,
            # "albedo_values": diffuse_values,
            "specular_values": specular_values,
            "image_paths": image_paths,
            "diffuse_paths": diffuse_paths,
            # "albedo_paths": diffuse_paths,
            "specular_paths": specular_paths
        }
        
        return example_dict
    
    return dataset, preprocess_psd, collate_fn_psd
        
        