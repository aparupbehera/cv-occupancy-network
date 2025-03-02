# Textured 3D Object Reconstruction Using Occupancy Networks  

## **Overview**  
This project enhances 3D reconstruction from 2D images by integrating feature extraction into Occupancy Networks. By leveraging a pre-trained convolutional neural network (CNN), the model refines geometric accuracy and texture quality, making it suitable for applications in AR/VR and digital content creation.  

## **Key Features**  
- **3D Reconstruction**: Generates high-fidelity 3D models from 2D images.  
- **Feature Extraction**: Uses CNNs to capture texture and object-specific features.  
- **Occupancy Networks**: Enhances implicit 3D representation with local feature integration.  
- **Evaluation Metrics**: Chamfer Distance & IoU for geometry, SSIM for texture quality.  
- **Datasets**: Trained and tested on ShapeNet and Pix3D.  

## **Installation**  
### **Dependencies**  
Ensure you have the following installed:  
- Python 3.x  
- PyTorch  
- TensorFlow  
- OpenCV  
- NumPy  
- Matplotlib  
- SciPy  

Install dependencies using:  
```bash
pip install -r requirements.txt
