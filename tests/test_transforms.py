from imax import transforms
from jax import numpy as jnp
from utils import compare


from jax import config
config.update("jax_debug_nans", True)

# test_img_rgba = jnp.asarray(Image.open('./test.jpeg').convert('RGBA')).astype('uint8')
# test_img_rgb = jnp.asarray(Image.open('./test.jpeg').convert('RGB')).astype('uint8')

rgb_img = jnp.array(
    [
        [[255, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ],
    dtype="float32",
)

rgba_img = jnp.array(
    [
        [[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ],
    dtype="float32",
)


def test_data():
    inputs = None
    targets = rgb_img
    outputs = rgba_img[:, :, :3]
    compare(inputs, targets, outputs)


def test_identity():
    inputs = rgba_img
    targets = rgba_img
    
    outputs = transforms.apply_transform(
        rgba_img,
        jnp.eye(4),
        mask_value=jnp.array([255., 255., 255., 255.]),
        bilinear=False,
    )
    compare(inputs, targets, outputs)

    outputs = transforms.apply_transform(
        rgba_img,
        jnp.eye(4),
        mask_value=jnp.array([255., 255., 255., 255.]),
        bilinear=True,
    )
    compare(inputs, targets, jnp.round(outputs))


def test_horizontal_flip():
    inputs = rgba_img
    targets = rgba_img[:, ::-1]
    
    outputs = transforms.apply_transform(
        rgba_img,
        transforms.flip(horizontal=True, vertical=False),
        mask_value=jnp.array([255., 255., 255., 255.]),
        bilinear=False,
    )
    compare(inputs, targets, outputs)

    outputs = transforms.apply_transform(
        rgba_img,
        transforms.flip(horizontal=True, vertical=False),
        mask_value=jnp.array([255., 255., 255., 255.]),
        bilinear=True,
    )
    compare(inputs, targets, jnp.round(outputs))


def test_vertical_flip():
    inputs = rgba_img
    targets = rgba_img[::-1]
    outputs = transforms.apply_transform(
        rgba_img, 
        transforms.flip(horizontal=False, vertical=True),
        mask_value=jnp.array([255, 255, 255, 255]),
        bilinear=False,
    )
    compare(inputs, targets, outputs)
    
    inputs = rgba_img
    targets = rgba_img[::-1]
    outputs = transforms.apply_transform(
        rgba_img, transforms.flip(horizontal=False, vertical=True),
        mask_value=jnp.array([255, 255, 255, 255]),
        bilinear=True,
    )
    compare(inputs, targets, jnp.round(outputs))


def test_rotate90():
    inputs = rgba_img
    targets = jnp.rot90(rgba_img, k=2)
    outputs = transforms.apply_transform(
        rgba_img,
        transforms.rotate90(n=2),
        mask_value=jnp.array([255, 255, 255, 255]),
        bilinear=False,
    )
    compare(inputs, targets, outputs)
    
    outputs = transforms.apply_transform(
        rgba_img,
        transforms.rotate90(n=2),
        mask_value=jnp.array([255, 255, 255, 255]),
        bilinear=True,
    )
    compare(inputs, targets, jnp.round(outputs))


def test_scale():
    factor = 3
    inputs = jnp.pad(
        jnp.ones((1, 1, 4), dtype="uint8") * 255,
        ((1, 1), (1, 1), (0, 0)),
        constant_values=0,
    )
    targets = inputs  # jnp.ones_like(rgba_img) * 255
    outputs = transforms.apply_transform(
        jnp.pad(
            jnp.ones((1, 1, 4), dtype="uint8"),
            ((1, 1), (1, 1), (0, 0)),
            constant_values=0,
        )
        * 255,
        transforms.scale_3d(scale_x=factor, scale_y=factor),
        mask_value=jnp.array([255, 255, 255, 255]),
        bilinear=False,
    )
    compare(inputs, targets, outputs)
