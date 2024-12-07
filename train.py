import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation
from occupancy_network import OccupancyNetwork, add_weight_decay
from dataset import SmallDataset
import os

def train_occupancy_network(data_dir, output_dir, num_epochs=100, batch_size=8, latent_dim=512, lr=1e-4):
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

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    weight_decay = 1e-5
    parameters = add_weight_decay(model, weight_decay)
    optimizer = optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, points, occupancies) in enumerate(train_loader):
            images, points, occupancies = images.to(device), points.to(device), occupancies.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, points)
            outputs = outputs.squeeze(-1)  # Remove extra dimension
            loss = model.compute_loss(outputs, occupancies)
            print(f"Loss: {loss.item()}, Outputs mean: {outputs.mean().item()}, Targets mean: {occupancies.mean().item()}")
            
            # Backward pass
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.grad.abs().mean()}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training Complete!")

# Example Usage
data_dir = './TrainingPairs'
output_dir = './Output'
train_occupancy_network(data_dir, output_dir)
