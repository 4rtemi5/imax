# imax
![tests](https://github.com/4rtemi5/imax/workflows/tests/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/imax.svg)](https://pypi.python.org/pypi/imax/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/112GaTnKDi-54eUIoXCREOZ_GuPAKNish?usp=sharing)

Fast and jittable Image augmentation library for Jax.

![sample_images](https://raw.githubusercontent.com/4rtemi5/imax/master/images/samples.png)

## Installation

```bash
pip install imax
```

## Usage

```python
from jax import random
import jax.numpy as jnp
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from imax import transforms, color_transforms, randaugment

# Setup
random_key = random.PRNGKey(0)
random_key, split_key = random.split(random_key)
image = jnp.asarray(Image.open('./test.png').convert('RGBA')).astype('uint8')

# Geometric transforms:
transform = transforms.rotate(rad=0.42)  # create transformation matrix
transformed_image = transforms.apply_transform(image, transform)   # apply transformation

# multiple transformations can be combined through matrix multiplication
# this makes multiple transforms much faster
double_transform = transform @ transform
twice_transformed_image = transforms.apply_transform(image, double_transform)

# Color transforms:
adjusted_image = color_transforms.posterize(image, bits=2)

# Randaugment:
randomized_image = randaugment.distort_image_with_randaugment(
    image,
    num_layers=3,   # number of random augmentations in sequence
    magnitude=10,   # magnitude of random augmentations
    random_key=split_key
)

# Show results:
results = [transformed_image, twice_transformed_image, adjusted_image, randomized_image]
fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 2),
                 axes_pad=0.1)

for ax, im in zip(grid, results):
    ax.axis('off')
    ax.imshow(im)
plt.show()
```