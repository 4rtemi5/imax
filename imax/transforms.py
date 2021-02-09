import jax.numpy as jnp
from imax.project import projective_inverse_warp

I = jnp.identity(4)


def scale(cx=1, cy=1, cz=1, c_=1):
    C = jnp.array([[1/cx,    0,    0,    0],
                   [   0, 1/cy,    0,    0],
                   [   0,    0, 1/cz,    0],
                   [   0,    0,    0, 1/c_]], dtype='float32')
    return lambda T=I: T @ C

def shear(sxy=0, sxz=0, syx=0, syz=0, szx=0, szy=0):
    S = jnp.array([[  1, sxy, sxz, 0],
                   [syx,   1, syz, 0],
                   [szx, szy,   1, 0],
                   [  0,   0,   0, 1]], dtype='float32')
    return lambda T=I: T @ S

def translate(tx=0, ty=0, tz=0):
    D = jnp.array([[1, 0, 0, tx],
                   [0, 1, 0, ty],
                   [0, 0, 1, tz],
                   [0, 0, 0, 1]], dtype='float32')
    return lambda T=I: T @ D

def flip(flip_horizontal=False, flip_vertical=False):
    rx = jnp.pi * flip_vertical
    ry = jnp.pi * flip_horizontal

    rcx = jnp.cos(rx)
    rsx = jnp.sin(rx)
    rx = jnp.array([[1, 0, 0, 0],
                    [0, rcx, rsx, 0],
                    [0, -rsx, rcx, 0],
                    [0, 0, 0, 1]])

    rcy = jnp.cos(ry)
    rsy = jnp.sin(ry)
    ry = jnp.array([[rcy, 0, -rsy, 0],
                    [0, 1, 0, 0],
                    [rsy, 0, rcy, 0],
                    [0, 0, 0, 1]])
    F = rx @ ry
    return lambda T=I: T @ F

def rotate90(n=0):
    rcz = jnp.cos(jnp.pi/2 * n)
    rsz = jnp.sin(jnp.pi/2 * n)
    R = jnp.array([[rcz, rsz, 0, 0],
                    [-rsz, rcz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return lambda T=I: T @ R

def rotate(rx=0, ry=0, rz=0):
    rcx = jnp.cos(rx)
    rsx = jnp.sin(rx)
    rx = jnp.array([[1, 0, 0, 0],
                    [0, rcx, rsx, 0],
                    [0, -rsx, rcx, 0],
                    [0, 0, 0, 1]])

    rcy = jnp.cos(ry)
    rsy = jnp.sin(ry)
    ry = jnp.array([[rcy, 0, -rsy, 0],
                    [0, 1, 0, 0],
                    [rsy, 0, rcy, 0],
                    [0, 0, 0, 1]])

    rcz = jnp.cos(rz)
    rsz = jnp.sin(rz)
    rz = jnp.array([[rcz, rsz, 0, 0],
                    [-rsz, rcz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R = rx @ ry @ rz
    return lambda T=I: T @ R

def apply_transforms(image,
                     transform,
                     mask_value=-1,
                     depth=-1,
                     intrinsic_matrix=-1):

    image = jnp.where(len(image.shape) == 3,
                      jnp.expand_dims(image, 0),
                      image)

    width = image.shape[-2]
    height = image.shape[-3]

    depth = jnp.where(jnp.any(depth >= 0),
                      depth,
                      jnp.expand_dims(jnp.ones(shape=(height, width)), 0))

    intrinsic_matrix = jnp.where(jnp.any(intrinsic_matrix >= 0),
                                 intrinsic_matrix,
                                 jnp.array([[[1, 0, width / 2],
                                             [0, 1, height / 2],
                                             [0, 0, 1]]],
                                           dtype='float32'))

    return projective_inverse_warp(image,
                                   transform,
                                   mask_value,
                                   intrinsic_matrix,
                                   depth)
