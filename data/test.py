import os
from PIL import Image
import numpy as np

dir = 'hole_5.png'
pil = Image.open(dir)
np_img = np.array(pil)
print(np_img.shape)