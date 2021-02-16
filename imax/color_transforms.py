# TODO: license
# ==============================================================================
"""
Color Transforms in Jax.
"""
import jax
from jax import lax, random
import jax.numpy as jnp


@jax.jit
def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

    Returns:
    A blended image Tensor of type uint8.
    """
    image_dtype = image1.dtype
    image1 = image1.astype('int32')
    image2 = image2.astype('int32')

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    return jnp.clip(temp, 0.0, 255.0).astype(image_dtype)


def get_random_cutout_mask(random_key, image_shape, max_mask_shape):
    """

    Args:
        random_key:
        image_shape:
        max_mask_shape:

    Returns:

    """
    # TODO: currently not jitable
    random_key, subkey = random.split(random_key)
    cutout_center_height = random.randint(subkey, shape=(), minval=0, maxval=image_shape[1])
    random_key, subkey = random.split(random_key)
    cutout_center_width = random.uniform(subkey, shape=(), minval=0, maxval=image_shape[0])
    random_key, subkey = random.split(random_key)
    pad_size_x = random.randint(subkey, shape=(), minval=0, maxval=max_mask_shape)
    random_key, subkey = random.split(random_key)
    pad_size_y = random.randint(subkey, shape=(), minval=0, maxval=max_mask_shape)

    lower_pad = jnp.maximum(0, cutout_center_height - pad_size_y).astype('int32')
    upper_pad = jnp.maximum(0, image_shape[0] - cutout_center_height - pad_size_y).astype('int32')
    left_pad = jnp.maximum(0, cutout_center_width - pad_size_x).astype('int32')
    right_pad = jnp.maximum(0, image_shape[1] - cutout_center_width - pad_size_x).astype('int32')

    mask = jnp.ones([(image_shape[0] - (lower_pad + upper_pad)),
                     (image_shape[1] - (left_pad + right_pad))])

    padding_dims = jnp.array([[lower_pad, upper_pad], [left_pad, right_pad]])
    mask = jnp.pad(
        mask,
        padding_dims, constant_values=0)
    mask = jnp.expand_dims(mask, -1)
    return mask.astype('bool')


# get_random_cutout_mask = jax.jit(get_random_cutout_mask, static_argnums=(2,))


@jax.jit
def cutout(image, mask, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
    image: An image Tensor of type uint8.
    mask: cutout mask to be applied to the image
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.

    Returns:
    An image Tensor that is of type uint8.
    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    mask = jnp.tile(mask, [1, 1, 3])
    image = jnp.where(
        mask,
        jnp.ones_like(image, dtype=image.dtype) * replace,
        image)

    if has_alpha:
        image = jnp.concatenate([image, alpha], axis=-1)

    return image


@jax.jit
def solarize(image, threshold=128):
    """

    Args:
        image:
        threshold:

    Returns:

    """
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = jnp.where(image < threshold, image, 255 - image)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def solarize_add(image, addition=0, threshold=128):
    """

    Args:
        image:
        addition:
        threshold:

    Returns:

    """
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    added_image = image.astype('int32') + addition
    added_image = jnp.clip(added_image, 0, 255).astype('uint8')
    degenerate = jnp.where(image < threshold, added_image, image)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def rgb_to_grayscale(rgb):
    """

    Args:
        rgb:

    Returns:

    """
    return jnp.dot(rgb[..., :3], jnp.array([0.2989, 0.5870, 0.1140]).astype('uint8'))


@jax.jit
def grayscale_to_rgb(grayscale):
    """

    Args:
        grayscale:

    Returns:

    """
    return jnp.stack((grayscale,) * 3, axis=-1)


@jax.jit
def color(image, factor):
    """
    Equivalent of PIL Color.
    Args:
        image:
        factor:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]
    degenerate = grayscale_to_rgb(rgb_to_grayscale(image))
    degenerate = blend(degenerate, image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def contrast(image, factor):
    """
    Equivalent of PIL Contrast.
    Args:
        image:
        factor:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = degenerate.astype('int32')

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist, _ = jnp.histogram(degenerate, bins=256, range=(0, 255))
    mean = jnp.sum(hist.astype('float32')) / 256.0
    degenerate = jnp.ones_like(degenerate, dtype='float32') * mean
    degenerate = jnp.clip(degenerate, 0.0, 255.0)
    degenerate = grayscale_to_rgb(degenerate).astype(image.dtype)
    degenerate = blend(degenerate, image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def brightness(image, factor):
    """
    Equivalent of PIL Brightness.
    Args:
        image:
        factor:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = jnp.zeros_like(image)
    degenerate = blend(degenerate, image, factor).astype(image.dtype)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def posterize(image, bits):
    """
    Equivalent of PIL Posterize.
    Args:
        image:
        bits:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    shift = 8 - bits
    degenerate = jnp.left_shift(jnp.right_shift(image, shift), shift)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate.astype('uint8')


@jax.jit
def autocontrast(image):
    """Implements Autocontrast function from PIL using Jax ops.
    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    @jax.jit
    def _scale_channel(_image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        low = jnp.min(_image).astype('float32')
        high = jnp.max(_image).astype('float32')

        # Scale the image, making the lowest value 0 and the highest value 255.
        @jax.jit
        def _scale_values(_im):
            scale = 255.0 / (high - low)
            offset = -low * scale
            _im = _im.astype('float32') * scale + offset
            _im = jnp.clip(_im, 0.0, 255.0)
            return _im.astype('uint8')

        return jnp.where(high > low,
                         _scale_values(_image),
                         _image)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    scaled_channel_1 = _scale_channel(image[:, :, 0])
    scaled_channel_2 = _scale_channel(image[:, :, 1])
    scaled_channel_3 = _scale_channel(image[:, :, 2])
    image = jnp.stack([scaled_channel_1, scaled_channel_2, scaled_channel_3], 2)

    if has_alpha:
        return jnp.concatenate([image, alpha], axis=-1)
    return image


@jax.jit
def sharpness(image, factor):
    """
    Implements Sharpness function from PIL using Jax ops.
    Args:
        image:
        factor:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    orig_image = image
    image = image.astype('float32')
    # Make image 4D for conv operation.
    image = jnp.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = jnp.array([[1, 1, 1],
                        [1, 5, 1],
                        [1, 1, 1]],
                       dtype='float32') / 13.
    kernel = jnp.reshape(kernel, (3, 3, 1, 1))
    # Tile across channel dimension.
    kernel = jnp.tile(kernel, [1, 1, 1, 3])
    # degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding='VALID', rate=[1, 1])
    degenerate = lax.conv_general_dilated(
        jnp.transpose(image, [0, 3, 1, 2]),    # lhs = NCHW image tensor
        jnp.transpose(kernel, [3, 2, 0, 1]),   # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        'VALID',  # padding mode
        feature_group_count=3)
    degenerate = jnp.clip(degenerate, 0.0, 255.0)
    degenerate = jnp.squeeze(degenerate.astype('uint8'), 0)
    degenerate = jnp.transpose(degenerate, [1, 2, 0])
    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = jnp.ones_like(degenerate)
    padded_mask = jnp.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = jnp.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = jnp.where(jnp.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    degenerate = blend(result, orig_image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


@jax.jit
def equalize(image):
    """
    Implements Equalize function from PIL using Jax ops.
    Args:
        image:

    Returns:

    """
    has_alpha = image.shape[-1] == 4

    @jax.jit
    def build_lut(histo, step):
        # Compute the cumulative sum, shifting by step // 2
        # and then normalization by step.
        lut = (jnp.cumsum(histo) + (step // 2)) // step
        # Shift lut, prepending with 0.
        lut = jnp.concatenate([jnp.array([0]), lut[:-1]], 0)
        # Clip the counts to be in range.  This is done
        # in the C code for image.point.
        return jnp.clip(lut, 0, 255)

    @jax.jit
    def scale_channel(img):
        """
        Scale the data in the channel to implement equalize.
        Args:
            img:

        Returns:

        """
        # im = im[:, :, c].astype('int32')
        img = img.astype('int32')
        # Compute the histogram of the image channel.
        histo = jnp.histogram(img, bins=255, range=(0, 255))[0]

        last_nonzero = jnp.argmax(histo[::-1] > 0)  # jnp.nonzero(histo)[0][-1]
        step = (jnp.sum(histo) - jnp.take(histo[::-1], last_nonzero)) // 255

        # if test_agains_original:
        #     # For the purposes of computing the step, filter out the nonzeros.
        #     nonzero = jnp.nonzero(histo)
        #     nonzero_histo = jnp.reshape(jnp.take(histo, nonzero), [-1])
        #     original_step = (jnp.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        #     assert step == original_step

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        return jnp.where(step == 0,
                         img.astype('uint8'),
                         jnp.take(build_lut(histo, step), img).astype('uint8'))

    scaled_channel_1 = scale_channel(image[:, :, 0])
    scaled_channel_2 = scale_channel(image[:, :, 1])
    scaled_channel_3 = scale_channel(image[:, :, 2])
    degenerate = jnp.stack([scaled_channel_1, scaled_channel_2, scaled_channel_3], 2)

    if has_alpha:
        return jnp.concatenate([degenerate, image[:, :, -1:]], axis=-1)
    return degenerate


@jax.jit
def invert(image):
    """
    Inverts the image pixels.
    Args:
        image:

    Returns:

    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = 255 - image

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate
