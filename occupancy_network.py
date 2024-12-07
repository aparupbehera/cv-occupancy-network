import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OccupancyNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(OccupancyNetwork, self).__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.point_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.occupancy_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, images, points):
        batch_size, num_points, _ = points.shape

        image_features = self.image_encoder(images)
        image_features = image_features.unsqueeze(1).expand(-1, num_points, -1)

        point_features = self.point_encoder(points.view(-1, 3)).view(batch_size, num_points, -1)

        combined_features = torch.cat([image_features, point_features], dim=-1)
        occupancies = self.occupancy_decoder(combined_features.view(-1, combined_features.size(-1)))
        occupancies = occupancies.view(batch_size, num_points, 1)
        occupancies = np.clip(occupancies, 1e-6, 1 - 1e-6)

        return occupancies

    def compute_loss(self, predictions, targets):
        # Weighted BCE Loss
        pos_weight = torch.tensor([10.0]).to(predictions.device)
        return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight)


# Add L2 regularization in the training loop
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
