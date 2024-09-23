import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

conf = get_config("zoedepth", "infer", pretrained_resource="local::/media/lab/Datasets/s1103539File/ZoeDepth-main/ZoeD_M12_N.pt")
model = build_model(conf).to(DEVICE)
print(model)