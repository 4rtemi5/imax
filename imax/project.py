# Original Source:
#  https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
#
# MIT License
#
# Copyright (c) 2017 Tinghui Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
'''
Image transformations with nearest neighbor and bilinear sampling.
'''
import jax
import jax.numpy as jnp


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.
    Args:
        depth: [batch, height, width]
        pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
        intrinsics: camera intrinsics [batch, 3, 3]
        is_homogeneous: return in homogeneous coordinates
    Returns:
        Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    height, width = depth.shape
    depth = jnp.reshape(depth, [1, -1])
    pixel_coords = jnp.reshape(pixel_coords, [3, -1])
    cam_coords = jnp.matmul(jnp.linalg.inv(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = jnp.ones([1, height * width])
        cam_coords = jnp.concatenate([cam_coords, ones], axis=0)
    cam_coords = jnp.reshape(cam_coords, [-1, height, width])
    return cam_coords


@jax.jit
def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.
    Args:
        cam_coords: [batch, 4, height, width]
        proj: [batch, 4, 4]
    Returns:
        Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    _, height, width = cam_coords.shape
    cam_coords = jnp.reshape(cam_coords, [4, -1])
    unnormalized_pixel_coords = jnp.matmul(proj, cam_coords)
    x_u = unnormalized_pixel_coords[0:1, :]  # [0:0:0, -1:1:-1]
    y_u = unnormalized_pixel_coords[1:2, :]  # [0:1:0, -1:1:-1]
    z_u = unnormalized_pixel_coords[2:3, :]  # [0:2:0, -1:1:-1]
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = jnp.concatenate([x_n, y_n], axis=0)
    pixel_coords = jnp.reshape(pixel_coords, [2, height, width])
    return jnp.transpose(pixel_coords, axes=[1, 2, 0])


def meshgrid(height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.
    Args:
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = jnp.matmul(jnp.ones(shape=[height, 1]),
                     jnp.transpose(jnp.expand_dims(
                        jnp.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = jnp.matmul(jnp.expand_dims(jnp.linspace(-1.0, 1.0, height), 1),
                     jnp.ones(shape=[1, width]))
    x_t = (x_t + 1.0) * 0.5 * jnp.array(width - 1, dtype='float32')
    y_t = (y_t + 1.0) * 0.5 * jnp.array(height - 1, dtype='float32')
    if is_homogeneous:
        ones = jnp.ones_like(x_t)
        coords = jnp.stack([x_t, y_t, ones], axis=0)
    else:
        coords = jnp.stack([x_t, y_t], axis=0)
    # coords = jnp.tile(jnp.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def compute_mask(coords_x, coords_y, x_max, y_max):
    """
    Computes invalid pixel coordinates.
    Args:
        coords_x: flattened vector of x coordinates
        coords_y: flattened vector of y coordinates
        x_max: maximum x coordinate
        y_max: maximum y coordinate

    Returns:
        vector of mask values
    """
    x_not_underflow = coords_x >= 0.0
    y_not_underflow = coords_y >= 0.0
    x_not_overflow = coords_x < x_max
    y_not_overflow = coords_y < y_max
    # z_positive = z > 0.0
    # x_not_nan = jnp.logical_not(jnp.isnan(coords_x))
    # y_not_nan = jnp.logical_not(jnp.isnan(coords_y))
    # not_nan = jnp.logical_and(x_not_nan, y_not_nan)
    # not_nan_mask = not_nan.astype('float32')
    # coords_x = tf.math.multiply_no_nan(coords_x, not_nan_mask)
    # coords_y = tf.math.multiply_no_nan(coords_y, not_nan_mask)
    mask_stack = jnp.stack([
        x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow,
        # z_positive,
        # not_nan
    ],
        axis=0)
    mask = jnp.all(mask_stack, axis=0)
    return mask


def projective_inverse_warp(img, transform, mask_value, intrinsics, depth, bilinear=True):
    """Inverse warp a source image to the target image plane based on projection.
    Args:
        img: the source image [batch, height_s, width_s, 3]
        transform: 4x4 transformation matrix
        mask_value: uint8 maks value of rgb/a mask value
        depth: depth map of the target image [batch, height_t, width_t]
        intrinsics: camera intrinsics [batch, 3, 3]
        bilinear: bool use bilinear or nearest sampling.
    Returns:
        Source image inverse warped to the target image plane [batch, height_t,
        width_t, 3]
    """
    height, width, _ = img.shape
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix
    filler = jnp.array([[0.0, 0.0, 0.0, 1.0]])
    # filler = jnp.tile(filler, [batch, 1, 1])
    intrinsics = jnp.concatenate([intrinsics, jnp.zeros([3, 1])], axis=1)
    intrinsics = jnp.concatenate([intrinsics, filler], axis=0)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = jnp.matmul(intrinsics, transform)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)

    output_img = jnp.where(bilinear,
                           bilinear_sampler(img, src_pixel_coords, mask_value),
                           nearest_sampler(img, src_pixel_coords, mask_value))
    return output_img.astype('uint8')


def nearest_sampler(imgs, coords, mask_value):
    """Construct a new image by nearest sampling from the input image.
    Points falling outside the source image boundary have value of mask_value.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
            width_t, 2]. height_t/width_t correspond to the dimensions of the output
            image (don't need to be the same as height_s/width_s). The two channels
            correspond to x and y coordinates respectively.
        mask_value: value of points outside of image. -1 for edge sampling.
        Returns:
            A new sampled image [height_t, width_t, channels]
    """
    coords_x, coords_y = jnp.split(coords, 2, axis=2)
    inp_size = imgs.shape
    out_size = list(coords.shape)
    out_size[2] = imgs.shape[2]

    coords_x = jnp.array(coords_x, dtype='float32')
    coords_y = jnp.array(coords_y, dtype='float32')

    y_max = jnp.array(jnp.shape(imgs)[0] - 1, dtype='float32')
    x_max = jnp.array(jnp.shape(imgs)[1] - 1, dtype='float32')
    zero = jnp.zeros([1], dtype='float32')
    eps = jnp.array([0.5], dtype='float32')

    coords_x_clipped = jnp.clip(coords_x, zero - eps, x_max + eps)
    coords_y_clipped = jnp.clip(coords_y, zero - eps, y_max + eps)

    x0 = jnp.round(coords_x_clipped)
    y0 = jnp.round(coords_y_clipped)

    x0_safe = jnp.clip(x0, zero, x_max)
    y0_safe = jnp.clip(y0, zero, y_max)

    # indices in the flat image to sample from
    dim2 = jnp.array(inp_size[1], dtype='float32')

    base_y0 = y0_safe * dim2
    idx00 = jnp.reshape(x0_safe + base_y0, [-1])

    # sample from imgs
    imgs_flat = jnp.reshape(imgs, [-1, inp_size[2]])
    imgs_flat = imgs_flat.astype('float32')
    output = jnp.reshape(jnp.take(imgs_flat, idx00.astype('int32'), axis=0), out_size)

    return jnp.where(jnp.any(mask_value > 0),
                     jnp.where(
                         compute_mask(coords_x, coords_y, x_max, y_max),
                         output,
                         jnp.ones_like(output) * jnp.reshape(jnp.array(mask_value), [1, 1, -1])
                     ),
                     output)


def bilinear_sampler(imgs, coords, mask_value):
    """Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value of mask_value.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
            width_t, 2]. height_t/width_t correspond to the dimensions of the output
            image (don't need to be the same as height_s/width_s). The two channels
            correspond to x and y coordinates respectively.
        mask_value: value of points outside of image. -1 for edge sampling.
        Returns:
            A new sampled image [height_t, width_t, channels]
    """
    coords_x, coords_y = jnp.split(coords, 2, axis=2)
    inp_size = imgs.shape
    out_size = list(coords.shape)
    out_size[2] = imgs.shape[2]

    coords_x = jnp.array(coords_x, dtype='float32')
    coords_y = jnp.array(coords_y, dtype='float32')

    y_max = jnp.array(jnp.shape(imgs)[0] - 1, dtype='float32')
    x_max = jnp.array(jnp.shape(imgs)[1] - 1, dtype='float32')
    zero = jnp.zeros([1], dtype='float32')
    eps = jnp.array([0.5], dtype='float32')

    coords_x_clipped = jnp.clip(coords_x, zero, x_max - eps)
    coords_y_clipped = jnp.clip(coords_y, zero, y_max - eps)

    x0 = jnp.floor(coords_x_clipped)
    x1 = x0 + 1
    y0 = jnp.floor(coords_y_clipped)
    y1 = y0 + 1

    x0_safe = jnp.clip(x0, zero, x_max)
    y0_safe = jnp.clip(y0, zero, y_max)
    x1_safe = jnp.clip(x1, zero, x_max)
    y1_safe = jnp.clip(y1, zero, y_max)

    # bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * jnp.equal(x0, x0_safe).astype('float32')
    # wt_x1 = (coords_x - x0) * jnp.equal(x1, x1_safe).astype('float32')
    # wt_y0 = (y1 - coords_y) * jnp.equal(y0, y0_safe).astype('float32')
    # wt_y1 = (coords_y - y0) * jnp.equal(y1, y1_safe).astype('float32')

    wt_x0 = x1_safe - coords_x  # 1
    wt_x1 = coords_x - x0_safe  # 0
    wt_y0 = y1_safe - coords_y  # 1
    wt_y1 = coords_y - y0_safe  # 0

    # indices in the flat image to sample from
    dim2 = jnp.array(inp_size[1], dtype='float32')

    base_y0 = y0_safe * dim2
    base_y1 = y1_safe * dim2
    idx00 = jnp.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    # sample from imgs
    imgs_flat = jnp.reshape(imgs, [-1, inp_size[2]])
    imgs_flat = imgs_flat.astype('float32')
    im00 = jnp.reshape(jnp.take(imgs_flat, idx00.astype('int32'), axis=0), out_size)
    im01 = jnp.reshape(jnp.take(imgs_flat, idx01.astype('int32'), axis=0), out_size)
    im10 = jnp.reshape(jnp.take(imgs_flat, idx10.astype('int32'), axis=0), out_size)
    im11 = jnp.reshape(jnp.take(imgs_flat, idx11.astype('int32'), axis=0), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = jnp.clip(jnp.round(w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11), 0, 255)

    # return output
    return jnp.where(jnp.all(mask_value >= 0),
                     jnp.where(
                         compute_mask(coords_x, coords_y, x_max, y_max),
                         output,
                         jnp.ones_like(output) * jnp.reshape(jnp.array(mask_value), [1, 1, -1])
                     ),
                     output)
