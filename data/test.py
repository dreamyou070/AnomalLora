import numpy as np

anomal_mask_np = np.array([[1,0,1],
                 [0,1,1],
                 [1,1,0]])
mask = np.repeat(np.expand_dims(anomal_mask_np, axis=2), 3, axis=2)
print(mask)