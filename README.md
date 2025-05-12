# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Build a Convolutional Autoencoder using PyTorch to remove noise from handwritten digit images in the MNIST dataset. The goal is to train the model to learn how to recover clean images from their noisy

## DESIGN STEPS

### STEP 1: 
Import all required libraries such as PyTorch, torchvision, and other helper modules for data loading, transformation, and visualization.

### STEP 2:
Download the MNIST dataset using torchvision.datasets, apply necessary transforms, and load it into DataLoader for batch processing.

### STEP 3:
Define a function to add Gaussian noise to the images to simulate real-world noisy input data.

### STEP 4: 
Build the Convolutional Autoencoder using PyTorch nn.Module with separate encoder and decoder sections using Conv2d and ConvTranspose2d.

### STEP 5: 
Initialize the autoencoder model, define the loss function as MSELoss, and select Adam as the optimizer.

### STEP 6:
Train the model over multiple epochs by feeding noisy images as input and computing the loss between the reconstructed output and the original clean image.

### STEP 7: 
Evaluate the model and visualize results by comparing original, noisy, and denoised images side by side.


## PROGRAM
### Name:KRITHIGA U
### Register Number:212223240076
```
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```

## OUTPUT

### Model Summary

![{15BB39DB-CEA3-4394-AB6C-8EB9C4A0E017}](https://github.com/user-attachments/assets/9a9cf647-e9ec-4266-ae9f-0e5ea8dfef60)


### Original vs Noisy Vs Reconstructed Image

![{42B4BB18-72E2-481A-BEC9-A6BE8D9A347D}](https://github.com/user-attachments/assets/75a55044-137c-4341-9e36-c90ff4478e72)

## RESULT

The trained autoencoder successfully removes noise from corrupted MNIST digits.
