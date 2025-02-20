import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
import torchmetrics
import torchmetrics.regression

class PointMe(pl.LightningModule):
    def __init__(self, teacher, student, config):
        """
        Args:
            teacher: teacher model
            student: student model
            config: configuration dictionary
        """
        super().__init__()
        self.teacher = teacher
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
        
        
        self.counting_loss = nn.MSELoss(reduction='mean')
        self.distillation_loss = nn.MSELoss(reduction='mean')

        self.train_mae = torchmetrics.regression.MeanAbsoluteError()
        self.val_mae = torchmetrics.regression.MeanAbsoluteError()
        self.test_mae = torchmetrics.regression.MeanAbsoluteError()

        self.train_mse = torchmetrics.regression.MeanSquaredError()
        self.val_mse = torchmetrics.regression.MeanSquaredError()
        self.test_mse = torchmetrics.regression.MeanSquaredError()
        
    def forward(self, batch, batch_idx):
        """
        Args:
            batch: input batch
        Returns:
            output: density map predictions
        """
        x = batch['image']
        teacher_features, student_features = self.forward_encoder(x)

        patch_indices = self.patch_locator(batch['points'], x.shape[-2:], self.config['patch_size'])

        # get the student features for each example
        examples = self.extract_patch_features(student_features, patch_indices)

        similarities = self.calculate_similarity(examples, student_features)

        enhanced_features = torch.cat([student_features, similarities], dim=1)
        output = self.forward_decoder(enhanced_features)

        return similarities, student_features, output



    def forward_encoder(self, x):
        """
        Args:
            x: input image
        Returns:
            teacher_features: teacher encoder features
            student_features: student encoder features
        """
        # teacher encoder
        with torch.no_grad():
            teacher_features = self.teacher.encoder(x)

        # student encoder
        student_features = self.student.encoder(x)

        return teacher_features, student_features
    
    def calculate_similarity(self, examples, features):
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
        dist_sq = dist_sq.view(dist_sq.size(0), dist_sq.size(1), features.size(2), features.size(3))

        # compute the similarity using a log based function
        similarities = torch.log((dist_sq + 1) / (dist_sq + 1e-4))

        return similarities


    def forward_decoder(self, enhanced_features):
        """
        Adapted from: https://github.com/abdumhmd/CounTX/blob/6e2af6403105984c2bd2a7aec45c517e1f535c76/models_counting_network.py#L135
        Args:
            similarities: similarity maps
        Returns:
            output: dot count predictions
        """
        density_map = F.interpolate(
            self.decoder_block1(enhanced_features),
            size = enhanced_features.shape[-1] * 2,
            mode = 'bilinear',
            align_corners = False,
        )

        density_map = F.interpolate(
            self.decoder_block2(density_map),
            size = enhanced_features.shape[-1] * 4,
            mode = 'bilinear',
            align_corners = False,
        )

        density_map = F.interpolate(
            self.decoder_block3(density_map),
            size = enhanced_features.shape[-1] * 8,
            mode = 'bilinear',
            align_corners = False,
        )

        density_map = F.interpolate(
            self.decoder_block4(density_map),
            size = enhanced_features.shape[-1] * 16,
            mode = 'bilinear',
            align_corners = False,
        )

        return density_map

    def patch_locator(self, points, img_size, patch_size):
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
            rounding_mode='floor',
        ).long()

        return patch_indices
    
    def extract_patch_features(self, features, patch_indices):
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