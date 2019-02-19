import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

file_path = os.path.join()
img = imread(file_path)  # shape: (H, W, 3), range: [0, 255]
# img = resize(img, (256, 256), mode='constant').astype(np.float32)  # (256, 256, 3), [0.0, 1.0]
plt.imshow(img)
plt.show()