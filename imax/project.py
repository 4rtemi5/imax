import jax
import jax.numpy as jnp


def getTransMatrix(trans_vec):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    batch_size = trans_vec.shape[0]
    # [B, 1, 1]
    one = jnp.ones([batch_size, 1, 1], dtype='float32')
    zero = jnp.zeros([batch_size, 1, 1], dtype='float32')

    T = jnp.concatenate([
        one, zero, zero, trans_vec[:, :, :1],
        zero, one, zero, trans_vec[:, :, 1:2],
        zero, zero, one, trans_vec[:, :, 2:3],
        zero, zero, zero, one
    ], axis=2)

    T = jnp.reshape(T, [batch_size, 4, 4])

    # T = tf.zeros([trans_vec.get_shape().as_list()[0],4,4],dtype=tf.float32)
    # for i in range(4):
    #     T[:,i,i] = 1
    # trans_vec = tf.reshape(trans_vec, [-1,3,1])
    # T[:,:3,3] = trans_vec
    return T


def rotFromAxisAngle(vec):
    """
    Convert axis angle into rotation matrix
    not euler angle but Axis
    :param vec: [B, 1, 3]
    :return:
    """
    angle = jnp.linalg.norm(vec, ord=2, axis=2, keepdims=True)
    axis = vec / (angle + 1e-7)

    ca = jnp.cos(angle)
    sa = jnp.sin(angle)

    C = 1 - ca

    x = axis[:, :, :1]
    y = axis[:, :, 1:2]
    z = axis[:, :, 2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # [B, 1, 1]
    one = jnp.ones_like(zxC, dtype='float32')
    zero = jnp.zeros_like(zxC, dtype='float32')

    rot_matrix = jnp.concatenate([
        x * xC + ca, xyC - zs, zxC + ys, zero,
        xyC + zs, y * yC + ca, yzC - xs, zero,
        zxC - ys, yzC + xs, z * zC + ca, zero,
        zero, zero, zero, one
    ], axis=2)

    rot_matrix = jnp.reshape(rot_matrix, [-1, 4, 4])

    # rot_matrix = tf.zeros([vec.get_shape().as_list()[0],4,4], dtype= tf.float32)
    #
    # rot_matrix[:, 0, 0] = tf.squeeze()
    # rot_matrix[:, 0, 1] = tf.squeeze()
    # rot_matrix[:, 0, 2] = tf.squeeze()
    # rot_matrix[:, 1, 0] = tf.squeeze()
    # rot_matrix[:, 1, 1] = tf.squeeze()
    # rot_matrix[:, 1, 2] = tf.squeeze()
    # rot_matrix[:, 2, 0] = tf.squeeze(zxC - ys)
    # rot_matrix[:, 2, 1] = tf.squeeze(yzC + xs)
    # rot_matrix[:, 2, 2] = tf.squeeze(z * zC + ca)
    # rot_matrix[:, 3, 3] = 1

    return rot_matrix


def pose_axis_angle_vec2mat(vec, invert=False):
    """
    Convert axis angle and translation into 4x4 matrix
    :param vec: [B,1,6] with former 3 vec is axis angle
    :return:
    """
    batch_size, _ = vec.shape

    axisvec = vec[:, 0:3]
    axisvec = jnp.reshape(axisvec, [batch_size, 1, 3])

    translation = vec[:, 3:6]
    translation = jnp.reshape(translation, [batch_size, 1, 3])

    R = rotFromAxisAngle(axisvec)

    if invert:
        R = jnp.transpose(R, [0, 2, 1])
        translation *= -1
    t = getTransMatrix(translation)

    if invert:
        M = jnp.matmul(R, t)
    else:
        M = jnp.matmul(t, R)
    return M


def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
       TODO: remove the dimension for 'N' (deprecated for converting all source
             poses altogether)
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = jnp.shape(z)[0]
    N = 1
    z = jnp.clip(z, -jnp.pi, jnp.pi)
    y = jnp.clip(y, -jnp.pi, jnp.pi)
    x = jnp.clip(x, -jnp.pi, jnp.pi)

    # Expand to B x N x 1 x 1
    z = jnp.expand_dims(jnp.expand_dims(z, -1), -1)
    y = jnp.expand_dims(jnp.expand_dims(y, -1), -1)
    x = jnp.expand_dims(jnp.expand_dims(x, -1), -1)

    zeros = jnp.zeros([B, N, 1, 1])
    ones = jnp.ones([B, N, 1, 1])

    cosz = jnp.cos(z)
    sinz = jnp.sin(z)
    rotz_1 = jnp.concatenate([cosz, -sinz, zeros], axis=3)
    rotz_2 = jnp.concatenate([sinz, cosz, zeros], axis=3)
    rotz_3 = jnp.concatenate([zeros, zeros, ones], axis=3)
    zmat = jnp.concatenate([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = jnp.cos(y)
    siny = jnp.sin(y)
    roty_1 = jnp.concatenate([cosy, zeros, siny], axis=3)
    roty_2 = jnp.concatenate([zeros, ones, zeros], axis=3)
    roty_3 = jnp.concatenate([-siny, zeros, cosy], axis=3)
    ymat = jnp.concatenate([roty_1, roty_2, roty_3], axis=2)

    cosx = jnp.cos(x)
    sinx = jnp.sin(x)
    rotx_1 = jnp.concatenate([ones, zeros, zeros], axis=3)
    rotx_2 = jnp.concatenate([zeros, cosx, -sinx], axis=3)
    rotx_3 = jnp.concatenate([zeros, sinx, cosx], axis=3)
    xmat = jnp.concatenate([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = jnp.matmul(jnp.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2mat(vec: jnp.array):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.shape
    translation = vec[:, 0:3]
    translation = jnp.expand_dims(translation, -1)
    rx = vec[:, 3:4]
    ry = vec[:, 4:5]
    rz = vec[:, 5:6]
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = jnp.squeeze(rot_mat, axis=1)
    filler = jnp.array([[[0.0, 0.0, 0.0, 1.0]]])
    filler = jnp.tile(filler, [batch_size, 1, 1])
    transform_mat = jnp.concatenate([rot_mat, translation], axis=2)
    transform_mat = jnp.concatenate([transform_mat, filler], axis=1)
    return transform_mat


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


def _repeat(x, n_repeats):
    rep = jnp.transpose(
        jnp.expand_dims(jnp.ones(shape=[n_repeats]), 1), [1, 0])
    rep = rep.astype('float32')
    x = jnp.matmul(jnp.reshape(x, (-1, 1)), rep)
    return jnp.reshape(x, [-1])


def compute_mask(coords_x, coords_y, x_max, y_max):
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
        depth: depth map of the target image [batch, height_t, width_t]
        pose: target to source camera transformation matrix [batch, 6], in the
              order of rx, ry, rz, tx, ty, tz
        intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
        Source image inverse warped to the target image plane [batch, height_t,
        width_t, 3]
    """
    height, width, channels = img.shape
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
    coords_x, coords_y = jnp.split(coords, 2, axis=2)
    inp_size = imgs.shape
    coord_size = coords.shape
    out_size = list(coords.shape)
    out_size[2] = imgs.shape[2]

    coords_x = jnp.array(coords_x, dtype='float32')
    coords_y = jnp.array(coords_y, dtype='float32')

    y_max = jnp.array(jnp.shape(imgs)[0] - 1, dtype='float32')
    x_max = jnp.array(jnp.shape(imgs)[1] - 1, dtype='float32')
    zero = jnp.zeros([1], dtype='float32')
    eps = jnp.array([0.5], dtype='float32')

    coords_x_clipped = jnp.clip(coords_x, -eps, x_max + eps)
    coords_y_clipped = jnp.clip(coords_y, -eps, y_max + eps)

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
        Returns:
            A new sampled image [batch, height_t, width_t, channels]
    """
    coords_x, coords_y = jnp.split(coords, 2, axis=2)
    inp_size = imgs.shape
    # coord_size = coords.shape
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
