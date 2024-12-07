import numpy as np
import torch
from skimage.measure import marching_cubes
import trimesh
from PIL import Image
from occupancy_network import OccupancyNetwork  # Import your model architecture

def preprocess_image(image_path, size=(128, 128)):
    """
    Preprocesses an image for the occupancy network.

    Args:
        image_path (str): Path to the input image.
        size (tuple): Target size for resizing the image.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, H, W].
    """
    with Image.open(image_path) as img:
        img = img.resize(size).convert("RGB")
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
    return img_tensor.unsqueeze(0)

def visualize_model_output(model, device, image_path, grid_size=32, threshold=0.5):
    """
    Visualizes the output of the occupancy network.

    Args:
        model: Trained occupancy network.
        device: Torch device (CPU or CUDA).
        image_path: Path to the input image.
        grid_size: Number of points per dimension in the 3D grid.
        threshold: Occupancy threshold for Marching Cubes.
    """
    image_tensor = preprocess_image(image_path).to(device)

    # Define the 3D grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        occupancies = model(image_tensor, grid_tensor).cpu().numpy().squeeze()

    volume = occupancies.reshape(grid_size, grid_size, grid_size)

    vertices, faces, normals, _ = marching_cubes(volume, level=threshold)
    vertices = (vertices / (grid_size - 1)) * 2 - 1

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()
    mesh.export('output_mesh.obj')
    print("Mesh saved to output_mesh.obj")

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 512
model = OccupancyNetwork(latent_dim)

# Load the saved model weights
checkpoint_path = "./Output/model_epoch_100.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
model.to(device)

# Visualize
image_path = "./RawData/teapot.png"
visualize_model_output(model, device, image_path)
