import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation
from occupancy_network import OccupancyNetwork
from dataset import SmallDataset
import os

def train_occupancy_network(data_dir, output_dir, num_epochs=100, batch_size=4, latent_dim=512, lr=1e-4):
    # Data Augmentation
    image_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
    ])

    # Dataset and DataLoader
    dataset = SmallDataset(data_dir, transform=image_transforms)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Optimizer, and Output Directory
    model = OccupancyNetwork(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, points, occupancies in train_loader:
            images, points, occupancies = images.to(device), points.to(device), occupancies.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, points)
            outputs = outputs.squeeze(-1)  # Remove extra dimension
            loss = model.compute_loss(outputs, occupancies)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training Complete!")

# Example Usage
data_dir = './TrainingPairs'
output_dir = './Output'
train_occupancy_network(data_dir, output_dir)
