import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Load depth image (grayscale)
depth_image = cv2.imread('output_image2.png', cv2.IMREAD_GRAYSCALE)

# Normalize depth values
normalized_depth = depth_image / np.max(depth_image)

# Apply colormap
colormap = cm.plasma(normalized_depth)[:, :, :3]  # Use only RGB channels
colormap = (colormap * 200).astype(np.uint8)  # Convert to 8-bit unsigned integer

# Display the result
plt.imshow(colormap)
plt.axis('off')
plt.show()
