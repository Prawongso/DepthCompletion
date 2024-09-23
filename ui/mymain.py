import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import colorize
import torch
import os
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Define output directory
output_dir = "myoutput"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function for depth estimation and saving output
def depth_estimation(image_path, output_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    img = Image.open(image_path)
    X = ToTensor()(img)
    X = X.unsqueeze(0).to(DEVICE)

    # Load model and move to correct device
    conf = get_config("zoedepth", "infer", pretrained_resource="local::ZoeDepth.pt")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Generate output
    with torch.no_grad():
        out = model.infer(X).cpu()

    # Colorize output and save
    pred = Image.fromarray(colorize(out))
    pred.save(output_path)

# Set input image and output paths
input_image_path = 'inpaintout\input_out_0.png'
output_image_path = os.path.join(output_dir, f"output_image.png")

# Run depth estimation and save result
depth_estimation(input_image_path, output_image_path)

print(f"Depth estimation completed and saved to {output_image_path}")
