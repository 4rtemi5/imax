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

"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
import jax
from jax import random
import jax.numpy as jnp
from imax import color_transforms
from imax import transforms

DEBUG = False

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

DEFAULT_RANDAUGMENT_VALUES = {
    # function_name -> probability
    # ORDER NEEDS TO BE KEPT THE SAME AS IN level_to_arg
    'AutoContrast':  1.,            # 0
    'Equalize':      1.,            # 1
    'Invert':        0.,            # 2
    'Posterize':     1.,            # 3
    'Solarize':      0.,            # 4
    'SolarizeAdd':   0.,            # 5
    'Color':         1.,            # 6
    'Contrast':      1.,            # 7
    'Brightness':    1.,            # 8
    'Sharpness':     1.,            # 9
    'Rotate':        1.,            # 10
    'ShearX':        1.,            # 11
    'ShearY':        1.,            # 12
    'TranslateX':    1.,            # 13
    'TranslateY':    1.,            # 14
    'FlipX':         1.,            # 15
    'FlipY':         1.,            # 16
    'Cutout':        1.,            # 17
}

DEFAULT_OPS = jnp.array(list(range(len(DEFAULT_RANDAUGMENT_VALUES.keys()))))
DEFAULT_PROBS = jnp.array(list(DEFAULT_RANDAUGMENT_VALUES.values())) / \
                sum(list(DEFAULT_RANDAUGMENT_VALUES.values()))


def level_to_arg(cutout_val, translate_val, negate, level, mask_value):
    """
    Translates the level to args for various functions.
    Args:
        cutout_val: value for cutout size of cutout function
        translate_val: value for
        negate: negate level
        level: input level

    Returns:

    """
    return tuple({
        'AutoContrast': (),
        'Equalize': (),
        'Invert': (),
        'Posterize': (5 - jnp.min(jnp.array([4, (level / _MAX_LEVEL * 4).astype('int32')])),),
        'Solarize': (((level / _MAX_LEVEL) * 256).astype('int32'),),
        'SolarizeAdd': (((level / _MAX_LEVEL) * 110).astype('int32'),),
        'Color': _enhance_level_to_arg(level),
        'Contrast': _enhance_level_to_arg(level),
        'Brightness': _enhance_level_to_arg(level),
        'Sharpness': _enhance_level_to_arg(level),
        'Rotate': (_rotate_level_to_arg(level, negate),),
        'ShearX': (_shear_level_to_arg(level, negate), 0),
        'ShearY': (0, _shear_level_to_arg(level, negate)),
        'TranslateX': (_translate_level_to_arg(translate_val, negate)[0], 0.),
        'TranslateY': (0., _translate_level_to_arg(translate_val, negate)[1]),
        'FlipX': (True, False),
        'FlipY': (False, True),
        'Cutout': (cutout_val, mask_value),
    }.values())


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""

    if level == 0:
        return 1.0  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return level


def _enhance_level_to_arg(level):
    return [(level / _MAX_LEVEL) * 1.8 + 0.1]


def _rotate_level_to_arg(level, negate):
    level = (level / _MAX_LEVEL) * jnp.pi
    level = jax.lax.cond(
        negate,
        lambda l: -l,
        lambda l: l,
        level
    )
    return level


def _shear_level_to_arg(level, negate):
    level = (level / _MAX_LEVEL)
    # Flip level to negative with 50% chance.
    level = jax.lax.cond(
        negate,
        lambda l: -l,
        lambda l: l,
        level
    )
    return level


def _translate_level_to_arg(translate_val, negate):
    # Flip level to negative with 50% chance.
    level = jax.lax.cond(
        negate,
        lambda t: (-t[0], -t[1]),
        lambda t: t,
        translate_val
    )
    return level


def _apply_ops(image, args, selected_op):
    """
    An abomination of a function to apply a chosen operation to an image.
    Args:
        image:
        args:
        selected_op:

    Returns:

    """
    geometric_transform = jnp.identity(4)
    image, geometric_transform = jax.lax.switch(selected_op, [
        lambda op: (color_transforms.autocontrast(op[0], *op[1][0]),
                    geometric_transform),  # 0
        lambda op: (color_transforms.equalize(op[0], *op[1][1]),
                    geometric_transform),  # 1
        lambda op: (color_transforms.invert(op[0], *op[1][2]),
                    geometric_transform),  # 2
        lambda op: (color_transforms.posterize(op[0], *op[1][3]).astype('uint8'),
                    geometric_transform),  # 3
        lambda op: (color_transforms.solarize(op[0], *op[1][4]),
                    geometric_transform),  # 4
        lambda op: (color_transforms.solarize_add(op[0], *op[1][5]),
                    geometric_transform),  # 5
        lambda op: (color_transforms.color(op[0], *op[1][6]),
                    geometric_transform),  # 6
        lambda op: (color_transforms.contrast(op[0], *op[1][7]),
                    geometric_transform),  # 7
        lambda op: (color_transforms.brightness(op[0], *op[1][8]),
                    geometric_transform),  # 8
        lambda op: (color_transforms.sharpness(op[0], *op[1][9]),
                    geometric_transform),  # 9
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.rotate(*op[1][10]))),  # 10
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.shear(*op[1][11]))),  # 11
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.shear(*op[1][12]))),  # 12
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.translate(*op[1][13]))),  # 13
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.translate(*op[1][14]))),  # 14
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.flip(*op[1][15]))),  # 15
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.flip(*op[1][16]))),  # 16
        lambda op: (color_transforms.cutout(op[0], *op[1][17]),
                    geometric_transform),  # 17
    ], (image, args))

    return image, geometric_transform


# @jax.jit
def _randaugment_inner_for_loop(_, in_args):
    """
    Loop body for for randougment.
    Args:
        i: loop iteration
        in_args: loop body arguments

    Returns:
        updated loop arguments
    """
    (image, geometric_transforms, random_key, available_ops, op_probs,
     magnitude, cutout_const, translate_const, join_transforms, default_replace_value) = in_args
    random_keys = random.split(random_key, num=8)
    random_key = random_keys[0]  # keep for next iteration
    op_to_select = random.choice(random_keys[1], available_ops, p=op_probs)
    mask_value = default_replace_value or random.randint(random_keys[2], [image.shape[-1]],
                                                         minval=-1, maxval=256)
    random_magnitude = random.uniform(random_keys[3], [], minval=0., maxval=magnitude)
    cutout_mask = color_transforms.get_random_cutout_mask(
        random_keys[4],
        image.shape,
        cutout_const)

    translate_vals = (random.uniform(random_keys[5], [], minval=0.0, maxval=1.0) * translate_const,
                      random.uniform(random_keys[6], [], minval=0.0, maxval=1.0) * translate_const)
    negate = random.randint(random_keys[7], [], minval=0, maxval=2).astype('bool')

    args = level_to_arg(cutout_mask, translate_vals, negate, random_magnitude, mask_value)

    if DEBUG:
        print(op_to_select, args[op_to_select])

    image, geometric_transform = _apply_ops(image, args, op_to_select)

    image, geometric_transform = jax.lax.cond(
        jnp.logical_or(join_transforms, jnp.all(
            jnp.not_equal(geometric_transform, jnp.identity(4)))),
        lambda op: (op[0], op[1]),
        lambda op: (transforms.apply_transform(op[0],
                                               op[1],
                                               mask_value=mask_value),
                    jnp.identity(4)),
        (image, geometric_transform)
    )

    geometric_transforms = jnp.matmul(geometric_transforms, geometric_transform)
    return(image, geometric_transforms, random_key, available_ops, op_probs,
           magnitude, cutout_const, translate_const, join_transforms, default_replace_value)


@jax.jit
def distort_image_with_randaugment(image,
                                   num_layers,
                                   magnitude,
                                   random_key,
                                   cutout_const=40,
                                   translate_const=50.0,
                                   default_replace_value=None,
                                   available_ops=DEFAULT_OPS,
                                   op_probs=DEFAULT_PROBS,
                                   join_transforms=False):
    """Applies the RandAugment policy to `image`.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        num_layers: Integer, the number of augmentation transformations to apply
          sequentially to an image. Represented as (N) in the paper. Usually best
          values will be in the range [1, 3].
        magnitude: Integer, shared magnitude across all augmentation operations.
          Represented as (M) in the paper. Usually best values are in the range
          [5, 30].
        random_key: random key to do random stuff
        join_transforms: reduce multiple transforms to one. Much more efficient but simpler.
        cutout_const: max cutout size int
        translate_const: maximum translation amount int
        default_replace_value: default replacement value for pixels outside of the image
        available_ops: available operations
        op_probs: probabilities of operations
        join_transforms: apply transformations immediately or join them

    Returns:
        The augmented version of `image`.
    """

    geometric_transforms = jnp.identity(4)

    for_i_args = (image, geometric_transforms, random_key, available_ops, op_probs,
                  magnitude, cutout_const, translate_const, join_transforms, default_replace_value)

    if DEBUG:  # un-jitted
        for i in range(num_layers):
            for_i_args = _randaugment_inner_for_loop(i, for_i_args)
    else:  # jitted
        for_i_args = jax.lax.fori_loop(0, num_layers, _randaugment_inner_for_loop, for_i_args)

    image, geometric_transforms = for_i_args[0], for_i_args[1]

    if join_transforms:
        replace_value = default_replace_value or random.randint(random_key,
                                                                [image.shape[-1]],
                                                                minval=0,
                                                                maxval=256)
        image = transforms.apply_transform(image, geometric_transforms, mask_value=replace_value)

    return image


if not DEBUG:
    distort_image_with_randaugment = jax.jit(distort_image_with_randaugment)
