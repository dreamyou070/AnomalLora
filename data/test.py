from PIL import Image
import numpy as np
import cv2
import numpy as np

rgb_img_dir = '000.png'
object_mask_dir = 'object_000.png'

rgb_np = np.array(Image.open(rgb_img_dir).convert('RGB').resize((256, 256)))
#mask = np.array(Image.open(object_mask_dir).convert('L').resize((256, 256)))
object_mask_np = np.array(Image.open(object_mask_dir).convert('RGB').resize((256, 256))) / 255
background_position = 1 - object_mask_np
background_img = Image.fromarray((rgb_np * background_position).astype(np.uint8)).convert('RGB')

