# TODO: license
"""
Geometric Transforms in Jax.
"""
import jax
import jax.numpy as jnp
from imax.project import projective_inverse_warp

I = jnp.identity(4)


@jax.jit
def scale_3d(scale_x=1., scale_y=1., scale_z=1., scale_xyz=1.):
    """
    Returns transformation matrix for 3d scaling.
    Args:
        scale_x: scaling factor in x-direction
        scale_y: scaling factor in y-direction
        scale_z: scaling factor in z-direction
        scale_xyz: scaling factor in all directions

    Returns:
        A 4x4 float32 transformation matrix.
    """
    matrix = jnp.array([[1 / scale_x, 0, 0, 0],
                        [0, 1 / scale_y, 0, 0],
                        [0, 0, 1 / scale_z, 0],
                        [0, 0, 0, 1 / scale_xyz]], dtype='float32')
    return matrix


@jax.jit
def scale(x_factor=1.0, y_factor=1.0):
    """
    Returns transformation matrix for 2d scaling.
    Args:
        x_factor:
        y_factor:

    Returns:

    """
    return scale_3d(scale_x=x_factor, scale_y=y_factor)


@jax.jit
def shear_3d(sxy=0., sxz=0., syx=0., syz=0., szx=0., szy=0.):
    """
    Returns transformation matrix for 3d shearing.
    Args:
        sxy:
        sxz:
        syx:
        syz:
        szx:
        szy:

    Returns:
        A 4x4 float32 transformation matrix.
    """
    matrix = jnp.array([[  1, sxy, sxz, 0],
                        [syx,   1, syz, 0],
                        [szx, szy,   1, 0],
                        [  0,   0,   0, 1]], dtype='float32')
    return matrix


@jax.jit
def shear(horizontal=0., vertical=0.):
    """
    Returns transformation matrix for 2d shearing.
    Args:
        horizontal:
        vertical:

    Returns:

    """
    return shear_3d(sxy=horizontal, syx=vertical)


@jax.jit
def translate_3d(translate_x=0, translate_y=0, translate_z=0):
    """
    Returns transformation matrix for 3d translation.
    Args:
        translate_x:
        translate_y:
        translate_z:

    Returns:
        A 4x4 float32 transformation matrix.
    """
    matrix = jnp.array([[1, 0, 0, translate_x],
                        [0, 1, 0, translate_y],
                        [0, 0, 1, translate_z],
                        [0, 0, 0, 1]], dtype='float32')
    return matrix


@jax.jit
def translate(horizontal, vertical):
    """
    Returns transformation matrix for 2d translation.
    Args:
        horizontal:
        vertical:

    Returns:

    """
    return translate_3d(translate_x=horizontal, translate_y=vertical)


@jax.jit
def flip(horizontal=False, vertical=False):
    """
    Returns transformation matrix for 2d flipping.
    Args:
        horizontal:
        vertical:

    Returns:

    """
    angle_x = jnp.pi * horizontal
    angle_y = jnp.pi * vertical

    rcx = jnp.cos(angle_x)
    rsx = jnp.sin(angle_x)
    rotation_y = jnp.array([[1,    0,   0, 0],
                            [0,  rcx, rsx, 0],
                            [0, -rsx, rcx, 0],
                            [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_x = jnp.array([[rcy, 0, -rsy, 0],
                            [  0, 1,    0, 0],
                            [rsy, 0,  rcy, 0],
                            [  0, 0,    0, 1]])
    matrix = rotation_x @ rotation_y
    return matrix


@jax.jit
def rotate90(n=0):
    """
    Returns transformation matrix for 2d rotation of multiples of 90°.
    Args:
        n:

    Returns:

    """
    rcz = jnp.cos(jnp.pi/2 * n)
    rsz = jnp.sin(jnp.pi/2 * n)
    matrix = jnp.array([[ rcz, rsz, 0, 0],
                        [-rsz, rcz, 0, 0],
                        [   0,   0, 1, 0],
                        [   0,   0, 0, 1]])
    return matrix


@jax.jit
def rotate_3d(angle_x=0, angle_y=0, angle_z=0):
    """
    Returns transformation matrix for 3d rotation.
    Args:
        angle_x:
        angle_y:
        angle_z:

    Returns:

    """
    rcx = jnp.cos(angle_x)
    rsx = jnp.sin(angle_x)
    rotation_x = jnp.array([[1,    0,   0, 0],
                    [0,  rcx, rsx, 0],
                    [0, -rsx, rcx, 0],
                    [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_y = jnp.array([[rcy, 0, -rsy, 0],
                    [  0, 1,    0, 0],
                    [rsy, 0,  rcy, 0],
                    [  0, 0,    0, 1]])

    rcz = jnp.cos(angle_z)
    rsz = jnp.sin(angle_z)
    rotation_z = jnp.array([[ rcz, rsz, 0, 0],
                    [-rsz, rcz, 0, 0],
                    [   0,   0, 1, 0],
                    [   0,   0, 0, 1]])
    matrix = rotation_x @ rotation_y @ rotation_z
    return matrix


@jax.jit
def rotate(rad):
    """
    Returns transformation matrix for 2d rotation around the z axis.
    Args:
        rad:

    Returns:

    """
    return rotate_3d(angle_z=rad)


@jax.jit
def apply_transform(image,
                    transform,
                    mask_value=-1,
                    depth=-1,
                    intrinsic_matrix=-1,
                    bilinear=True):
    """
    Applies a 3d transformation to an image. Can deal with depth data and intrinsic matrices.
    Args:
        image:
        transform:
        mask_value:
        depth:
        intrinsic_matrix:
        bilinear:

    Returns:

    """

    width = image.shape[1]
    height = image.shape[0]

    depth = jnp.where(jnp.any(depth >= 0),
                      depth,
                      jnp.ones(shape=(height, width)))

    intrinsic_matrix = jnp.where(jnp.any(intrinsic_matrix >= 0),
                                 intrinsic_matrix,
                                 jnp.array([[1, 0, (width - 1) / 2.],
                                            [0, 1, (height - 1) / 2.],
                                            [0, 0, 1]],
                                           dtype='float32'))

    return projective_inverse_warp(image,
                                   transform,
                                   mask_value,
                                   intrinsic_matrix,
                                   depth,
                                   bilinear=bilinear)
