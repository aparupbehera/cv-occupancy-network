from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SmallDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(data_path)
        image = data['image']  # Shape: [H, W, 3] for RGB images
        point_cloud = data['point_cloud']  # Shape: [num_points, 3]
        occupancies = (point_cloud[:, -1] > 0).astype(np.float32)  # Last column as target

        # Convert to tensor and rearrange dimensions for PyTorch
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # [3, H, W]

        return image, \
               torch.tensor(point_cloud[:, :3], dtype=torch.float32), \
               torch.tensor(occupancies, dtype=torch.float32)
