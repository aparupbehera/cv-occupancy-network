import torch
import torch.nn as nn
import torchvision.models as models

class OccupancyNetwork(nn.Module):
    def __init__(self, latent_dim=512):
        super(OccupancyNetwork, self).__init__()
        # Encoder: Pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.fc_latent = nn.Linear(resnet.fc.in_features, latent_dim)

        # Decoder: MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, 256),  # Concatenate latent vector with 3D points
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Occupancy probability
            nn.Sigmoid()
        )

    def forward(self, image, points):
        # Encode image to latent vector
        latent = self.encoder(image).view(image.size(0), -1)
        latent = self.fc_latent(latent)

        # Repeat latent vector for each point
        latent_repeated = latent.unsqueeze(1).expand(-1, points.size(1), -1)
        inputs = torch.cat([latent_repeated, points], dim=-1)

        # Decode to occupancy probabilities
        occupancies = self.decoder(inputs)
        return occupancies

    def compute_loss(self, outputs, targets):
        # Binary cross-entropy loss for occupancy predictions
        criterion = nn.BCELoss()
        return criterion(outputs, targets)
