from imgaug import augmenters as iaa
import numpy as np
rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
a = np.zeros((512,512), dtype=np.uint8)
a = rot(image=a)