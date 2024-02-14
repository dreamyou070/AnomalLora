import numpy as np

perlin_noise = np.array([[0.8,0,0.3],
                         [0.1,0.5,0.9],])
noise = np.where(perlin_noise > 0.5, perlin_noise, 0)
