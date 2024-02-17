import os
from PIL import Image
import numpy as np
data_dir = '226.png'
image = Image.open(data_dir)
np_img = np.array(image)
h, w = image.size
for h_index in range(h) :
    for w_index in range(w) :
        if h_index < int(h/10) or h_index > int(h*9/10) :
            np_img[h_index,w_index] = 0
for w_index in range(w) :
    for h_index in range(h) :
        if w_index < int(w/90) or w_index > int(w*89/90) :
            np_img[h_index,w_index] = 0
image = Image.fromarray(np_img)
image.save('226_mask.png')

