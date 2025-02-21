import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import h5py
import pandas as pd

class CellDataset(Dataset):
    '''
        A custom dataset for end-to-end training of PointMe model.
        Args:
            - img_list: list of image paths
            - augment: bool, whether to apply data augmentation or not (these are to be applied on input image, density map, cell locations and embeddings)

        Returns:
            - img: torch.Tensor, input image
            - density_map: torch.Tensor, density map
            - cell_locs: torch.Tensor, cell locations
            - embeddings: torch.Tensor, embeddings

    '''

    def __init__(self, img_list, augment=False):
        self.img_list = img_list
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        
        dmap = h5py.File(img_path.replace('.png', '.h5').replace('images', 'densities'), 'r')
        density_map = np.array(dmap['density'])
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

        cell_locs = pd.read_csv(img_path.replace('.png', '.csv').replace('images', 'points'))[['X', 'Y']].values
        cell_locs = torch.tensor(cell_locs, dtype=torch.float32).unsqueeze(0)

        embeddings = torch.load(img_path.replace('.png', '.npy').replace('images', 'embeddings'), map_location='cpu')
        print(embeddings.shape)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        if self.augment:
            # Apply data augmentation here
            pass
        

        return {"image": img, "density_map": density_map, "cell_locs": cell_locs, "embeddings": embeddings}
    

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
