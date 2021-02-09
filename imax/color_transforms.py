from jax import lax, random
import jax.numpy as jnp


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
    if factor == 0.0:
        return jnp.array(image1)
    if factor == 1.0:
        return jnp.array(image2)

    image1 = image1.astype('float32')
    image2 = image2.astype('float32')

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.astype('uint8')

    # Extrapolate:
    #
    # We need to clip and then cast.
    return jnp.clip(temp, 0.0, 255.0).astype('uint8')


def cutout(image, pad_size, random_key, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    random_key: jax random key
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
    image_height = jnp.shape(image)[0]
    image_width = jnp.shape(image)[1]

    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:,:,:3], image[:,:,-1:]

    # Sample the center location in the image where the zero mask will be applied.
    random_key, subkey = random.split(random_key)
    cutout_center_height = random.uniform(
        random_key, shape=(), minval=0, maxval=image_height).astype('int32')

    cutout_center_width = random.uniform(
        random_key, shape=(), minval=0, maxval=image_height).astype('int32')

    lower_pad = jnp.maximum(0, cutout_center_height - pad_size)
    upper_pad = jnp.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = jnp.maximum(0, cutout_center_width - pad_size)
    right_pad = jnp.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = jnp.pad(
        jnp.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = jnp.expand_dims(mask, -1)
    mask = jnp.tile(mask, [1, 1, 3])
    image = jnp.where(
        jnp.equal(mask, 0),
        jnp.ones_like(image, dtype=image.dtype) * replace,
        image)

    if has_alpha:
        image = jnp.concatenate([image, alpha], axis=-1)

    return image


def solarize(image, threshold=128):
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


def solarize_add(image, addition=0, threshold=128):
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


def rgb_to_grayscale(rgb):
    return jnp.dot(rgb[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))


def grayscale_to_rgb(grayscale):
    return jnp.stack((grayscale,) * 3, axis=-1)


def color(image, factor):
    """Equivalent of PIL Color."""
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]
    degenerate = grayscale_to_rgb(rgb_to_grayscale(image))
    degenerate = blend(degenerate, image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


def contrast(image, factor):
    """Equivalent of PIL Contrast."""
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
    degenerate = grayscale_to_rgb(degenerate.astype('uint8'))
    degenerate = blend(degenerate, image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = jnp.zeros_like(image)
    degenerate = blend(degenerate, image, factor)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    shift = 8 - bits
    degenerate = jnp.left_shift(jnp.right_shift(image, shift), shift)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
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

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = jnp.min(image).astype('float32')
        hi = jnp.max(image).astype('float32')

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype('float32') * scale + offset
            im = jnp.clip(im, 0.0, 255.0)
            return im.astype('uint8')

        return jnp.where(hi > lo,
                         scale_values(image),
                         image)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = jnp.stack([s1, s2, s3], 2)

    if has_alpha:
        return jnp.concatenate([image, alpha], axis=-1)
    return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
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
                        [1, 1, 1]], dtype='float32')
    kernel = jnp.reshape(kernel, (3, 3, 1, 1)) / 13.
    # Tile across channel dimension.
    kernel = jnp.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    # degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding='VALID', rate=[1, 1])
    degenerate = lax.conv(jnp.transpose(image,[0,3,1,2]),    # lhs = NCHW image tensor
                          jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
                          (1, 1),  # window strides
                          'VALID') # padding mode
    degenerate = jnp.clip(degenerate, 0.0, 255.0)
    degenerate = jnp.squeeze(degenerate.astype('uint8'), 0)

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


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].astype('int32')
        # Compute the histogram of the image channel.
        histo, _ = jnp.histogram(im, bins=255, range=(0, 255))

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = jnp.where(jnp.not_equal(histo, 0))
        nonzero_histo = jnp.reshape(jnp.take(histo, nonzero), [-1])
        step = (jnp.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (jnp.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = jnp.concatenate([jnp.array([0]), lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return jnp.clip(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if jnp.equal(step, 0):
            return im.astype('unit8')
        return jnp.take(build_lut(histo, step), im).astype('uint8')

    # TODO: Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    degenerate = jnp.stack([s1, s2, s3], 2)

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate


def invert(image):
    """Inverts the image pixels."""
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = 255 - image

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate
