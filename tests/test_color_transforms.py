from imax import color_transforms
from jax import numpy as jnp
from utils import compare


im1 = jnp.ones((3, 3, 3), dtype='float32') * 255
im2 = jnp.zeros((3, 3, 3), dtype='float32')

mask = jnp.array([
        [[0], [0], [0]],
        [[0], [1], [0]],
        [[0], [0], [0]]],
        dtype='bool')


def test_blend():
    factor = jnp.array(0.55 * 255, dtype='float32')
    inputs = [im1, im2]
    targets = (im2 * factor).astype('float32')
    outputs = color_transforms.blend(inputs[0], inputs[1], factor)
    compare(inputs, targets, outputs)


def test_cutout():
    inputs = im2
    targets = mask.astype('float32') * 42
    outputs = color_transforms.cutout(inputs, mask, replace=42)
    compare(inputs, targets, outputs)


def test_solarize():
    inputs = im1
    targets = im2
    outputs = color_transforms.solarize(inputs, threshold=128)
    compare(inputs, targets, outputs)


def test_solarize_add():
    inputs = im1 * mask
    targets = jnp.ones_like(im1) * 245 * mask + 10
    outputs = color_transforms.solarize_add(inputs, addition=10, threshold=128)
    compare(inputs, targets, outputs)


def test_gray_to_rgb_to_gray():
    inputs = jnp.ones_like(im1)
