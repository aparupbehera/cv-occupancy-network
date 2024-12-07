import os
from PIL import Image
import numpy as np
import trimesh

def preprocess_images_to_rgb(image_dir, output_dir, size=(128, 128)):
    """
    Preprocesses images: resizes, retains RGB channels, and normalizes them.

    Args:
        image_dir (str): Directory containing raw PNG images.
        output_dir (str): Directory to save preprocessed images.
        size (tuple): Target size for resizing (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            filepath = os.path.join(image_dir, filename)
            try:
                # Open image, resize, and normalize
                with Image.open(filepath) as img:
                    img = img.resize(size).convert("RGB")  # Ensure RGB mode
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]

                    # Save the preprocessed image as a .npy file
                    output_path = os.path.join(output_dir, filename.replace(".png", ".npy"))
                    np.save(output_path, img_array)
                    print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def preprocess_meshes(mesh_dir, output_dir, num_points=2048):
    """
    Converts 3D meshes to point clouds.

    Args:
        mesh_dir (str): Directory containing OBJ files.
        output_dir (str): Directory to save point cloud data.
        num_points (int): Number of points to sample from each mesh.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(mesh_dir):
        if filename.endswith(".obj"):
            filepath = os.path.join(mesh_dir, filename)
            try:
                # Load mesh and sample points
                mesh = trimesh.load(filepath)
                point_cloud = mesh.sample(num_points)

                # Save as .npy
                output_path = os.path.join(output_dir, filename.replace(".obj", ".npy"))
                np.save(output_path, point_cloud)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def create_training_pairs(image_dir, mesh_dir, output_dir):
    """
    Creates training pairs of images and point clouds.

    Args:
        image_dir (str): Directory containing preprocessed images (.npy).
        mesh_dir (str): Directory containing preprocessed point clouds (.npy).
        output_dir (str): Directory to save paired data as .npz files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = sorted(os.listdir(image_dir))
    mesh_files = sorted(os.listdir(mesh_dir))

    for image_file, mesh_file in zip(image_files, mesh_files):
        if image_file.replace(".npy", "") == mesh_file.replace(".npy", ""):
            image_path = os.path.join(image_dir, image_file)
            mesh_path = os.path.join(mesh_dir, mesh_file)

            # Load data
            image_data = np.load(image_path)
            mesh_data = np.load(mesh_path)

            # Save paired data
            output_path = os.path.join(output_dir, image_file.replace(".npy", ".npz"))
            np.savez(output_path, image=image_data, point_cloud=mesh_data)
            print(f"Created training pair: {output_path}")
        else:
            print(f"Mismatch: {image_file} and {mesh_file}")

image_dir = "./RawData"
image_output_dir = "./PreprocessedImages"
preprocess_images_to_rgb(image_dir, image_output_dir)

mesh_dir = "./RawData"
mesh_output_dir = "./PreprocessedMeshes"
preprocess_meshes(mesh_dir, mesh_output_dir)

training_pair_dir = "./TrainingPairs"
create_training_pairs(image_output_dir, mesh_output_dir, training_pair_dir)

