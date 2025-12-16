# Autoencoder 
Currently this repository implements a **Convolutional Autoencoder (CAE)** using PyTorch to learn compact latent representations of handwritten digits from the **MNIST dataset**. Beyond reconstruction, the project focuses on **interpreting and visualizing the learned latent space** using feature map visualization and t-SNE.

## 🔍 Overview
Autoencoders are unsupervised neural networks used for representation learning and dimensionality reduction.  

Here, we:
- Train a convolutional autoencoder on MNIST
- Visualize encoder feature maps for individual samples
- Analyze the latent space using **t-SNE**
- Compare original images, latent activations, and reconstructions across all digit classes

## ⚡ Model Architecture
### Encoder
- Convolutional layers with ReLU activation
- Batch normalization
- Max pooling for spatial downsampling

### Decoder
- Upsampling layers
- Convolutional layers with ReLU
- Final Sigmoid activation for pixel normalization

The model outputs:
- Reconstructed image
- Latent representation

## 🏋️ Training Details
- Dataset: MNIST (training split)
- Loss Function: Reconstruction loss (MSE)
- Optimizer: Adam
- Epochs: 5
- Batch-wise training with shuffled data

## 📊 Visualizations Included
### Encoder Feature Maps
Visual inspection of the 32 convolutional feature maps produced by the encoder for a single digit.

### Latent Space (t-SNE)
- 10,000 latent vectors projected into 2D
- Color-coded by digit label
- Demonstrates clustering and separability of digit representations

### End-to-End Comparison
For each digit (0–9), the following are displayed:
- Original image
- Averaged latent activation map
- Reconstructed image

## 🛠️ Tech Stack
- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn


## 🚀 Future Improvements
- Add variational autoencoder (VAE) comparison
- Explore clustering metrics in latent space
- Extend to other datasets (Fashion-MNIST, CIFAR-10)

## 📄 License

This project is open-source and available for educational and research purposes.
