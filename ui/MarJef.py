import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import colorize
import torch
import cv2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

io = 6
def process_image(image_path):
    # Load the image
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image)

    # Perform the Fourier Transform
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a high-pass filter
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with center square (low frequencies) set to zero
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the central square
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    # Apply the high-pass filter
    f_transform_shifted_filtered = f_transform_shifted * mask

    # Inverse Fourier Transform
    f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
    image_filtered_array = np.fft.ifft2(f_transform_filtered)
    image_filtered_array = np.abs(image_filtered_array)

    # Combine the original image with the high-pass filtered image
    alpha = 0.5  # Scaling factor for the high-pass filtered image
    image_combined = image_array + alpha * image_filtered_array

    # Clip the values to be in the range [0, 255] and convert to uint8
    image_combined = np.clip(image_combined, 0, 255).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(image_combined, threshold1=100, threshold2=200)
    edges = abs(edges*100000)
    image_combined = image_combined + alpha * edges
    image_combined = np.clip(image_combined, 0, 255).astype(np.uint8)

    return image_array, image_filtered_array, edges, image_combined

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load image
img = Image.open(f'pic{io}.jpg')
orig_size = img.size
X = ToTensor()(img)
X = X.unsqueeze(0).to(DEVICE)

# Load model and move to correct device
conf = get_config("zoedepth", "infer", pretrained_resource="local::ZoeDepth.ptt")
model = build_model(conf).to(DEVICE)
model.eval()

# Generate output
with torch.no_grad():
    out = model.infer(X).cpu()

# Colorize output
pred = Image.fromarray(colorize(out))
pred.save(f"save\output_image{io}.png")

# Process image and get intermediate results
original, filtered, edges, image_combined = process_image(f"save\output_image{io}.png")

# Plot the original, high-pass filtered, Canny edge detection, and combined images
plt.figure(figsize=(18, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# High-Pass Filtered Image
plt.subplot(1, 4, 2)
plt.imshow(filtered, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.axis('off')

# Canny edge detection
plt.subplot(1, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

# Combined Image
plt.subplot(1, 4, 4)
plt.imshow(image_combined, cmap='gray')
plt.title('Combined Image')
plt.axis('off')

plt.show()
#python -m ui.app