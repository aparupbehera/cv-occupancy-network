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


def visualize_model_output(model, device, image_path, grid_size=64, threshold=0.5):
    image_tensor = preprocess_image(image_path).to(device)

    # Save preprocessed image for inspection
    from torchvision.utils import save_image
    save_image(image_tensor, 'preprocessed_image.png')

    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    grid = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Grid tensor shape: {grid_tensor.shape}")

    with torch.no_grad():
        occupancies = model(image_tensor, grid_tensor).detach().cpu().numpy().squeeze()
        print(f"Occupancies shape: {occupancies.shape}")
        print(f"Occupancies min: {occupancies.min()}, max: {occupancies.max()}")
        print(f"Occupancies mean: {occupancies.mean()}, std: {occupancies.std()}")
        # Visualize raw occupancies
        import matplotlib.pyplot as plt
        # Visualize raw occupancies
        plt.imshow(occupancies.reshape(grid_size, grid_size, grid_size).sum(axis=2))
        plt.savefig('occupancy_slice.png')

        volume = occupancies.reshape(grid_size, grid_size, grid_size)

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            try:
                vertices, faces, normals, _ = marching_cubes(volume, level=threshold)
                print(f"Marching cubes successful with threshold {threshold}")
                vertices = (vertices / (grid_size - 1)) * 2 - 1
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                mesh.show()
                mesh.export('output_mesh.obj')
                print("Mesh saved to output_mesh.obj")
            except ValueError as e:
                print(f"Error in marching cubes: {e}")
                print(f"Marching cubes failed with threshold {threshold}: {e}")
                print("Try adjusting the threshold or check if the occupancy predictions are valid.")

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 512
model = OccupancyNetwork(latent_dim)

# Load the saved model weights
checkpoint_path = "./Output/model_epoch_100.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Checkpoint loaded from {checkpoint_path}")
model.eval()
model.to(device)

# Visualize
image_path = "./RawData/teapot.png"
visualize_model_output(model, device, image_path)
