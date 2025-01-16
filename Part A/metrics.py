import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def calculate_inception_score_and_fid(
    generated_samples,
    real_samples,
    batch_size=32,
    device=None,
    resize=True
):
    """
    Calculates Inception Score (IS) and Fréchet Inception Distance (FID) for generated samples.

    Args:
        generated_samples (torch.Tensor): Generated images, shape (N, 3, H, W), values in [0, 1].
        real_samples (torch.Tensor): Real images for FID comparison, shape (N, 3, H, W), values in [0, 1].
        batch_size (int): Batch size for processing images.
        device (torch.device): Device to run computations on. If None, uses CUDA if available.
        resize (bool): Whether to resize images to 299x299 required by InceptionV3.

    Returns:
        is_mean (float): The Inception Score mean.
        is_std (float): The Inception Score standard deviation.
        fid_score (float): The Fréchet Inception Distance.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Resize((299, 299)) if resize else nn.Identity()

    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    # Function to preprocess and convert images
    def preprocess(batch):
        # Resize if necessary
        if resize:
            batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale to [0, 255] and convert to uint8
        batch = (batch * 255).clamp(0, 255).to(torch.uint8)
        return batch

    # Process generated samples for IS and FID
    for i in range(0, generated_samples.size(0), batch_size):
        batch = generated_samples[i:i + batch_size]
        batch = preprocess(batch).to(device)
        fid.update(batch, real=False)
        is_metric.update(batch)

    # Process real samples for FID
    for i in range(0, real_samples.size(0), batch_size):
        batch = real_samples[i:i + batch_size]
        batch = preprocess(batch).to(device)
        fid.update(batch, real=True)

    # Compute scores
    fid_score = fid.compute().item()
    is_mean, is_std = is_metric.compute()

    return is_mean, is_std, fid_score

# Example Usage
if __name__ == "__main__":
    # Ensure you have torchmetrics installed
    # You can install it via: pip install torchmetrics

    import torch

    # Assuming you have generated_samples and real_samples as torch.Tensor
    # For demonstration, let's create random tensors
    N = 1000  # Number of samples
    generated_samples = torch.rand(N, 3, 64, 64)  # Example generated images in [0, 1]
    real_samples = torch.rand(N, 3, 64, 64)       # Example real images in [0, 1]

    is_mean, is_std, fid_score = calculate_inception_score_and_fid(
        generated_samples,
        real_samples,
        batch_size=32,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        resize=True
    )

    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"FID Score: {fid_score:.4f}")
