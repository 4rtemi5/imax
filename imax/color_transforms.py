# Original Source:
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Color Transforms in Jax.
"""

from functools import partial
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


@jax.jit
def _expand_vertical(_, mask):
    kernel = jnp.ones((1, 1, 3, 1))
    mask = lax.conv(mask,  # lhs = NCHW image tensor
                    kernel,  # rhs = OIHW conv kernel tensor
                    (1, 1),  # window strides
                    'SAME')
    return mask


@jax.jit
def _expand_horizontal(_, mask):
    kernel = jnp.ones((1, 1, 1, 3))
    mask = lax.conv(mask,  # lhs = NCHW image tensor
                    kernel,  # rhs = OIHW conv kernel tensor
                    (1, 1),  # window strides
                    'SAME')
    return mask


@partial(jax.jit, static_argnums=(1,))
def get_random_cutout_mask(random_key, image_shape, mask_size):
    """
    Creates a random cutout mask.
    Args:
        random_key: jax.random key
        image_shape: desired 2d mask shape
        mask_size: maximum cutout height/width

    Returns:

    """
    # Workaround over unjitable approach
    random_keys = random.split(random_key, 4)
    random_key, subkeys = random_keys[0], random_keys[1:]
    cutout_height = random.randint(subkeys[0], shape=(), minval=1, maxval=mask_size).astype('int32')
    cutout_width = random.uniform(subkeys[1], shape=(), minval=1, maxval=mask_size).astype('int32')
    mask = random.uniform(subkeys[2], (image_shape[0], image_shape[1]))
    mask = (mask == jnp.max(mask)).astype('float32')
    mask = jnp.expand_dims(mask, axis=(0, -1))
    mask = jnp.transpose(mask, [0, 3, 1, 2])
    mask = jax.lax.fori_loop(0, cutout_height - 1, _expand_vertical, mask)
    mask = jax.lax.fori_loop(0, cutout_width - 1, _expand_horizontal, mask)
    mask = jnp.transpose((mask > 0.0)[0], [1, 2, 0])
    return mask


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
    replace = replace.astype('uint8')
    replace_has_alpha = not len(replace.shape) == 0 and replace.shape[-1] == 4
    alpha = None
    if len(replace.shape) == 0 or replace.shape[-1] == 1:
        replace = jnp.tile(jnp.array(replace), (3,))

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    mask_3 = jnp.tile(mask, [1, 1, 3])

    image = jnp.where(
        mask_3,
        jnp.ones_like(image, dtype=image.dtype) * replace[:3],
        image)

    if replace_has_alpha:
        alpha = jnp.where(
            mask,
            jnp.ones_like(alpha, dtype=image.dtype) * replace[3:],
            alpha
        )

    if has_alpha:
        image = jnp.concatenate([image, alpha], axis=-1)

    return image


@jax.jit
def solarize(image, threshold=128):
    """
    Solarize augmentation.
    Args:
        image: input image (uint8)
        threshold: solarize threshold (int)

    Returns:
        Augmented image.
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
        image: input image (uint8)
        addition: addition value (int)
        threshold: solarize threshold (int)

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
    Transforms rgb image to grayscale.
    Args:
        rgb: rgb image

    Returns:
        Grayscale image.
    """
    return jnp.dot(rgb[..., :3], jnp.array([0.2989, 0.5870, 0.1140]).astype('uint8'))


@jax.jit
def grayscale_to_rgb(grayscale):
    """
    Transforms grayscale image to three channel image.
    Args:
        grayscale: single channes grayscale image.

    Returns:
        Three channel image.
    """
    return jnp.stack((grayscale,) * 3, axis=-1)


@jax.jit
def color(image, factor):
    """
    Equivalent of PIL Color.
    Args:
        image: image tensor
        factor: float factor

    Returns:
         Augmented image.
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
        image: image tensor
        factor: float factor

    Returns:
        Augmented image
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
        image: image tensor
        factor: float factor

    Returns:
        Augmented image.
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
        image: image tensor
        bits: bits to shift

    Returns:
        Augmented image.
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
        image: image tensor
        factor: float factor

    Returns:
        Augmented image.
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
        image: image tensor

    Returns:
        Augmented image.
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
            img: channel to scale.

        Returns:
            scaled channel
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
        image: image tensor

    Returns:
        Augmented image.
    """
    has_alpha = image.shape[-1] == 4
    alpha = None

    if has_alpha:
        image, alpha = image[:, :, :3], image[:, :, -1:]

    degenerate = 255 - image

    if has_alpha:
        return jnp.concatenate([degenerate, alpha], axis=-1)
    return degenerate
