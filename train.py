import torch

import lightning as pl
from torch.utils.data import DataLoader, random_split
from data_handling.cell_dataset import DistillationDataset, CountingDataset
from modeling import PointMe, TinyViT
from build_sam import sam_model_registry
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train():
    # Define the dataset
    img_paths = glob('../Datasets/PanNuke_multi_count/embeddings/*.npy')
    img_paths = [img.replace('.npy', '.png').replace('embeddings', 'images') for img in img_paths]

    print(f"Found {len(img_paths)} images")

    all_dataset = DistillationDataset(img_paths)

    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size

    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Define the model
    config = {
        "stage" : "distillation",
        "distill_coef": 1,
        "lr": 1e-2,
        "patch_size": 16,
        "image_size": 256,
    }

    student = sam_model_registry["student_encoder"]()

    model = PointMe(student, config)

    print("Successfully loaded the model")

    # Define the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices = 1,
        max_epochs=100,
        logger=None
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()




