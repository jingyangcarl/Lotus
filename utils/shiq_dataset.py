
import os
from datasets import Dataset as Dataset_hf
from tqdm.rich import tqdm
from PIL import Image
from torchvision import transforms
import torch

# class SHQI10825(Dataset):
#     """
#     SHQI10825 dataset.
#     """

#     def __init__(self, data_dir: str, split: str = 'train'):
#         super().__init__(data_dir)
#         self.data_dir = data_dir
#         self.data = self.load_data()

#     def load_data(self):
#         # Implement the logic to load the SHQI10825 dataset
#         pass

#     def __getitem__(self, index):
#         # Implement the logic to get an item from the dataset
#         pass

#     def __len__(self):
#         return len(self.data)

def get_SHIQ10825_dataset(data_dir, split='train'):
    
    split_dir = os.path.join(data_dir, split)
    
    # load data and construct the dataset
    data_dict = {
        "image": [], # 
        "diffuse": [],
        "specular": [],
    }
    file_count = sum(len(files) for _, _, files in os.walk(split_dir))  # Get the number of files
    print(f"Total files in SHIQ dataset at {split_dir}: {file_count}")
    with tqdm(total=file_count, desc=f"Walking through SHIQ dataset") as pbar:
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                pbar.update(1)
                if file.endswith("_A.png"): 
                    image_path = os.path.join(root, file)
                    diffuse_path = image_path.replace("_A.png", "_D.png")
                    specular_path = image_path.replace("_A.png", "_S.png")
                    # specular_label_path = static_path.replace("_A.png", "_T.png")
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
    
    
    def preprocess_shiq10825(examples):
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
    
    
    def collate_fn_shiq10825(examples):
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
            # "diffuse_values": diffuse_values,
            "albedo_values": diffuse_values,
            "specular_values": specular_values,
            "image_paths": image_paths,
            # "diffuse_paths": diffuse_paths,
            "albedo_paths": diffuse_paths,
            "specular_paths": specular_paths
        }
        
        return example_dict
    
    return dataset, preprocess_shiq10825, collate_fn_shiq10825
        
        