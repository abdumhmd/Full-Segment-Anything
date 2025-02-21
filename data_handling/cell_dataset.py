import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DistillationDataset(Dataset):
    '''
        A custom dataset for distillation of PointMe model.

        Args:
            - img_list: list of image paths
            
        Returns:
            - A dictionary containing the following
                - image: input image
                - embeddings: embeddings from SAM-H model
    '''
    def __init__(self, img_list):
        self.img_list = img_list
        self.transform = A.Compose([
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        
        embeddings = torch.load(img_path.replace('.png', '.npy').replace('images', 'embeddings'))
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        return {"image": img, "embeddings": embeddings}
    
class CountingDataset(Dataset):
    '''
        A custom dataset for counting the number of cells in an image.

        Args:
            - img_list: list of image paths
            - augmentations: Albumentation augmentations to be applied on the images, points and density map
            
        Returns:
            - A dictionary containing the following
                - image: input image
                - cell_locs: locations of prompt points
                - density_map: density map of the image
    '''
    def __init__(self, img_list, augmentations=None):
        self.img_list = img_list
        self.augmentations = augmentations


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        
        dmap = h5py.File(img_path.replace('.png', '.h5').replace('images', 'densities'), 'r')
        density_map = np.array(dmap['density'])
        density_map = torch.tensor(density_map, dtype=torch.float32)

        cell_locs = pd.read_csv(img_path.replace('.png', '.csv').replace('images', 'points'))[['X', 'Y']].values
        cell_locs = torch.tensor(cell_locs, dtype=torch.float32)

        if self.augmentations:
            augmented = self.augmentations(image=img, keypoints=cell_locs, mask=density_map)
            img = augmented['image']
            cell_locs = augmented['keypoints']
            density_map = augmented['mask']

        return {"image": img, "cell_locs": cell_locs, "density_map": density_map}
    

        


''' Testing the dataset
from glob import glob
import matplotlib.pyplot as plt
import cv2
import random

img_list = glob('../Datasets/Cell Datasets/PanNuke_multi_count/points/*.csv')

print('Number of images:', len(img_list))

img_list = [img.replace('.csv', '.png').replace('points', 'images') for img in img_list]

dataset = CellDataset(img_list, augment=True)

example = random.choice(dataset)

img = example['image']
img = img.squeeze(0).permute(1, 2, 0).numpy()
density_map = example['density_map']
cell_locs = example['cell_locs']
embeddings = example['embeddings']

print('Image shape:', img.shape)
print('Density map shape:', density_map.shape)
print('Cell locations shape:', cell_locs.shape)
print('Embeddings shape:', embeddings.shape)

plt.figure(figsize=(5, 5))
plt.subplot(1,3,1)
plt.imshow(img)
plt.axis('off')
plt.title('Input Image')

plt.subplot(1,3,2)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for cell in cell_locs.squeeze(0):
    x, y = cell
    cv2.circle(img_bgr, (int(x), int(y)), 3, (245, 247, 73), -1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Prompts')

    

plt.subplot(1,3,3)
plt.imshow(density_map.squeeze(0).squeeze(0))
plt.title(f"{density_map.sum().item():.0f} Cells")
plt.axis('off')
plt.show()
'''
