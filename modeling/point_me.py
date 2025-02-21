import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
import torchmetrics
import torchmetrics.regression


class PointMe(pl.LightningModule):
    def __init__(self, student, config):
        """
        Args:
            teacher: teacher model
            student: student model
            config: configuration dictionary
        """
        super().__init__()
        self.student = student

        self.config = config

        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(259, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        self.counting_loss = nn.MSELoss(reduction="mean")
        self.distillation_loss = nn.MSELoss(reduction="mean")

        self.train_mae = torchmetrics.regression.MeanAbsoluteError()
        self.val_mae = torchmetrics.regression.MeanAbsoluteError()
        self.test_mae = torchmetrics.regression.MeanAbsoluteError()

        self.train_mse = torchmetrics.regression.MeanSquaredError()
        self.val_mse = torchmetrics.regression.MeanSquaredError()
        self.test_mse = torchmetrics.regression.MeanSquaredError()

    def forward(self, batch):
        """
        Args:
            batch: input batch
        Returns:
            output: density map predictions
        """
        x = batch["image"]
        student_features = self._forward_encoder(x)

        if self.config["stage"] == "distillation":
            return student_features

        patch_indices = self._patch_locator(
            batch["cell_locs"], x.shape[-2:], self.config["patch_size"]
        )

        # get the student features for each example
        examples = self._extract_patch_features(
            student_features, patch_indices
        )

        similarities = self._calculate_similarity(examples, student_features)

        enhanced_features = torch.cat([student_features, similarities], dim=1)
        dmap = self._forward_decoder(enhanced_features)

        return similarities, dmap

    def _forward_encoder(self, x):
        """
        Extract features from the student model
        Args:
            x: input image
        Returns:
            student_features: student encoder features
        """
        # student encoder
        student_features = self.student(x)

        return student_features

    def _calculate_similarity(self, examples, features):
        """
        Args:
            examples: Example features based on dots
            features: Features from the student model
        Returns:
            similarity: similarity maps for each example
        """
        # flatten the features (N, C, H, W) -> (N, C, H * W)
        features_flat = features.view(features.size(0), features.size(1), -1)

        # expand the examples (N, K, C) -> (N, K, C, 1)
        examples_expanded = examples.unsqueeze(-1)

        # expand the features (N, C, H * W) -> (N, C, 1, H * W)
        features_expanded = features_flat.unsqueeze(2)

        # compute the squared L2 distance between examples and features
        dist_sq = torch.sum((examples_expanded - features_expanded) ** 2, dim=1)

        # reshape the distance (N, K, H * W) -> (N, K, H, W)
        dist_sq = dist_sq.view(
            dist_sq.size(0), dist_sq.size(1), features.size(2), features.size(3)
        )

        # compute the similarity using a log based function
        similarities = torch.log((dist_sq + 1) / (dist_sq + 1e-4))

        return similarities

    def _forward_decoder(self, enhanced_features):
        """
        Adapted from: https://github.com/abdumhmd/CounTX/blob/6e2af6403105984c2bd2a7aec45c517e1f535c76/models_counting_network.py#L135
        Args:
            similarities: similarity maps
        Returns:
            output: dot count predictions
        """
        density_map = F.interpolate(
            self.decoder_block1(enhanced_features),
            size=enhanced_features.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        density_map = F.interpolate(
            self.decoder_block2(density_map),
            size=enhanced_features.shape[-1] * 4,
            mode="bilinear",
            align_corners=False,
        )

        density_map = F.interpolate(
            self.decoder_block3(density_map),
            size=enhanced_features.shape[-1] * 8,
            mode="bilinear",
            align_corners=False,
        )

        density_map = F.interpolate(
            self.decoder_block4(density_map),
            size=enhanced_features.shape[-1] * 16,
            mode="bilinear",
            align_corners=False,
        )

        return density_map

    def _patch_locator(self, points, img_size, patch_size):
        """
        Map batched points to patches
        Args:
            points: Tensor of shape (N, K, 2) containing K points for each of the N images
            img_size: Tuple containing the image size (H, W)
            patch_size: size of each patch (E.g. 16 for ViT-Base)
        Returns:
            patch_indices: Tensor of shape (N, K, 2) with patch indices (row, col)
        """
        assert points.dim() == 3 and points.shape[-1] == 2, "Invalid shape for points"

        if isinstance(patch_size, int):
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size

        # clamp coordinates to image size
        h, w = img_size
        points_clamped = points.clone()
        points_clamped[..., 0] = torch.clamp(points_clamped[..., 0], 0, h - 1)
        points_clamped[..., 1] = torch.clamp(points_clamped[..., 1], 0, w - 1)

        # calculate patch indices
        patch_indices = torch.div(
            points_clamped,
            torch.tensor([ph, pw], device=points.device, dtype=torch.float32),
            rounding_mode="floor",
        ).long()

        return patch_indices

    def _extract_patch_features(self, features, patch_indices):
        """
        Extract features from patch embeddings using the given patch indices.

        Args:
            features : Feature tensor of shape (bs, 256, 16, 16).
            patch_indices: Tensor of shape (bs, num_points, 2) containing (row, col) indices.

        Returns:
            Extracted features of shape (bs, num_points, 256).
        """
        bs, _ = patch_indices.shape[:2]

        row_indices = patch_indices[..., 0]
        col_indices = patch_indices[..., 1]

        features = features[
            torch.arange(bs).unsqueeze(1),
            :,
            row_indices,
            col_indices,
        ]
        return features.permute(0, 2, 1)

    def training_step(self, batch, batch_idx):
        

        if self.config['stage'] == 'distillation':
            student_features = self(batch)
            loss = self.config["distill_coef"] * self.distillation_loss(
                student_features, batch["embeddings"]
            )
        else:
            _, dmap = self(batch)
            loss = self.config["count_coef"] * self.counting_loss(
                dmap, batch["density_map"]
            ) 

            self.train_mae(dmap, batch["density_map"])
            self.train_mse(dmap, batch["density_map"])
            self.log("train_mae", self.train_mae, on_epoch=True, on_step=True, prog_bar=True)
            self.log("train_mse", self.train_mse, on_epoch=True, on_step=True, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
    
        if self.config['stage'] == 'distillation':
            student_features = self(batch)
            loss = self.config["distill_coef"] * self.distillation_loss(
                student_features, batch["embeddings"]
            )
        else:
            _, dmap = self(batch)
            loss = self.config["count_coef"] * self.counting_loss(
                dmap, batch["density_map"]
            )

            self.val_mae(dmap, batch["density_map"])
            self.val_mse(dmap, batch["density_map"])
            self.log("val_mae", self.val_mae, on_epoch=True, prog_bar=True)
            self.log("val_mse", self.val_mse, on_epoch=True, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
    
        if self.config['stage'] == 'distillation':
            student_features = self(batch)
            loss = self.config["distill_coef"] * self.distillation_loss(
                student_features, batch["embeddings"]
            )
        else:
            _, dmap = self(batch)
            loss = self.config["count_coef"] * self.counting_loss(
                dmap, batch["density_map"]
            )

            self.test_mae(dmap, batch["density_map"])
            self.test_mse(dmap, batch["density_map"])
            self.log("test_mae", self.test_mae, on_epoch=True, prog_bar=True)
            self.log("test_mse", self.test_mse, on_epoch=True, prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


""" Testing the model
# from tiny_vit import TinyViT


# student = TinyViT(

#         img_size=1024,
#         in_chans=3,
#         num_classes=1000,
#         embed_dims=[64, 128, 160, 320],
#         depths=[2, 2, 6, 2],
#         num_heads=[2, 4, 5, 10],
#         window_sizes=[7, 7, 14, 7],
#         mlp_ratio=4.,
#         drop_rate=0.,
#         drop_path_rate=0.0,
#         use_checkpoint=False,
#         mbconv_expand_ratio=4.0,
#         local_conv_size=3,
#         layer_lr_decay=0.8
#     )



# image_encoder = TinyViT(
#         img_size=1024,
#         in_chans=3,
#         num_classes=1000,
#         embed_dims=[64, 128, 160, 320],
#         depths=[2, 2, 6, 2],
#         num_heads=[2, 4, 5, 10],
#         window_sizes=[7, 7, 14, 7],
#         mlp_ratio=4.,
#         drop_rate=0.,
#         drop_path_rate=0.0,
#         use_checkpoint=False,
#         mbconv_expand_ratio=4.0,
#         local_conv_size=3,
#         layer_lr_decay=0.8
#     )

# config = {
#     'patch_size': 16
# }

# model = PointMe(image_encoder, student, config)

# # random input
# x = torch.randn(2, 3, 256, 256)

# # random points 3 points for each image
# points = torch.randint(0, 256, (2, 3, 2))

# teacher_features, student_features = model.forward_encoder(x)
# print(f"Teacher features: {teacher_features.shape}")
# print(f"Student features: {student_features.shape}")

# patch_indices = model.patch_locator(points, x.shape[-2:], config['patch_size'])
# print(f"Patch indices: {patch_indices.shape}")

# examples = model.extract_patch_features(student_features, patch_indices)
# print(f"Examples: {examples.shape}")

# similarities = model.calculate_similarity(examples, student_features)
# print(f"Similarities: {similarities.shape}")

# enhanced_features = torch.cat([student_features, similarities], dim=1)
# output = model.forward_decoder(enhanced_features)
# print(f"Output: {output.shape}")

"""
