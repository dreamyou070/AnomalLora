from PIL import Image
import numpy as np
import cv2
import numpy as np


mask_dir = 'banded_0002.jpg'
pil_img = Image.open(mask_dir)
org_h, org_w = pil_img.size

black_image = np.zeros((org_h, org_w, 3), dtype=np.uint8)
black_pil = Image.fromarray(black_image)
for i in range(1,10) :
    black_pil.save(f'black_00{i}.jpg')
