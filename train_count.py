import torch

import lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback

from torch.utils.data import DataLoader, random_split
from data_handling.cell_dataset import DistillationDataset, CountingDataset
from modeling import PointMe, TinyViT
from build_sam import sam_model_registry
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train():
    # Define the dataset
    # Define the dataset
    img_paths = glob("../Datasets/Cell Datasets/PanNuke_multi_count/embeddings/*.npy")
    img_paths = [
        img.replace(".npy", ".png").replace("embeddings", "images") for img in img_paths
    ]

    print(f"Found {len(img_paths)} images")

    train_size = int(0.8 * len(img_paths))

    train_imgs = img_paths[:train_size]
    val_imgs = img_paths[train_size:]

    train_transforms = A.Compose(
        [
            # A.RandomCrop(width=192, height=192),
            A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # rotate to 90, 180, 270
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    val_transforms = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256, border_mode=0),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    train_dataset = CountingDataset(train_imgs, augmentations=train_transforms)
    val_dataset = CountingDataset(val_imgs, augmentations=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    config = {
        "stage": "counting",
        "lr": 1e-3,
        "patch_size": 16,
        "image_size": 256,
        "count_coef": 1,
    }

    student = sam_model_registry["student_encoder"]()

    model = PointMe.load_from_checkpoint(
        checkpoint_path='epoch=58-step=65490.ckpt',
        student=student,
        config=config,
    )

    # Freeze the student encoder
    model._freeze_student()

    print("Successfully loaded the model")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="pointme-counting-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )


    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=2,
        logger=None,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    train()