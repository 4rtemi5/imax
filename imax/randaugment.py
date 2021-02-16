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
    'Cutout':        float(DEBUG),  # 15
}

DEFAULT_OPS = jnp.array(list(range(len(DEFAULT_RANDAUGMENT_VALUES.keys()))))
DEFAULT_PROBS = jnp.array(list(DEFAULT_RANDAUGMENT_VALUES.values())) / \
                sum(list(DEFAULT_RANDAUGMENT_VALUES.values()))


def level_to_arg(cutout_val, translate_val, negate, level):
    """
    Translates the level to args for various functions.
    Args:
        cutout_val: value for cutout size of cutout function
        translate_val: value for
        negate:
        level

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
        'Cutout': (cutout_val,),   # Not working currently
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


# def _parse_policy_info(name, prob, level, cutout_val, translate_val, negate):
#     """Return the function that corresponds to `name` and update `level` param."""
#     func, is_geometric_transform = NAME_TO_FUNC[name]
#     args = level_to_arg(cutout_val, translate_val, negate, level)[name]
#     return func, is_geometric_transform, prob, args


# def _apply_func_with_prob(func, image, args, prob):
#     """Apply `func` to image w/ `args` as input with probability `prob`."""
#     assert isinstance(args, tuple)
#
#     # If prob is a function argument, then this randomness is being handled
#     # inside the function, so make sure it is always called.
#     # pytype:disable=wrong-arg-types
#     if 'prob' in inspect.getargspec(func)[0]:
#         prob = 1.0
#     # pytype:enable=wrong-arg-types
#
#     # Apply the function with probability `prob`.
#     should_apply_op = tf.cast(
#         tf.floor(tf.random_uniform([], dtype=tf.float32) + prob), tf.bool)
#     augmented_image = tf.cond(
#         should_apply_op,
#         lambda: func(image, *args),
#         lambda: image)
#     return augmented_image


# def select_and_apply_random_policy(policies, image):
#     """Select a random policy from `policies` and apply it to `image`."""
#     policy_to_select = tf.random_uniform([], maxval=len(policies), dtype=tf.int32)
#     # Note that using tf.case instead of tf.conds would result in significantly
#     # larger graphs and would even break export for some larger policies.
#     for (i, policy) in enumerate(policies):
#         image = tf.cond(
#             tf.equal(i, policy_to_select),
#             lambda selected_policy=policy: selected_policy(image),
#             lambda: image)
#     return image


# def build_and_apply_nas_policy(policies, image,
#                                augmentation_hparams):
#     """Build a policy from the given policies passed in and apply to image.
#
#   Args:
#     policies: list of lists of tuples in the form `(func, prob, level)`, `func`
#       is a string name of the augmentation function, `prob` is the probability
#       of applying the `func` operation, `level` is the input argument for
#       `func`.
#     image: tf.Tensor that the resulting policy will be applied to.
#     augmentation_hparams: Hparams associated with the NAS learned policy.
#
#   Returns:
#     A version of image that now has data augmentation applied to it based on
#     the `policies` pass into the function.
#   """
#     replace_value = [128, 128, 128]
#
#     # func is the string name of the augmentation function, prob is the
#     # probability of applying the operation and level is the parameter associated
#     # with the tf op.
#
#     # tf_policies are functions that take in an image and return an augmented
#     # image.
#     tf_policies = []
#     for policy in policies:
#         tf_policy = []
#         # Link string name to the correct python function and make sure the correct
#         # argument is passed into that function.
#         for policy_info in policy:
#             policy_info = list(policy_info) + [replace_value, augmentation_hparams]
#
#             tf_policy.append(_parse_policy_info(*policy_info))
#
#         # Now build the tf policy that will apply the augmentation procedue
#         # on image.
#         def make_final_policy(tf_policy_):
#             def final_policy(image_):
#                 for func, prob, args in tf_policy_:
#                     image_ = _apply_func_with_prob(
#                         func, image_, args, prob)
#                 return image_
#
#             return final_policy
#
#         tf_policies.append(make_final_policy(tf_policy))
#
#     augmented_image = select_and_apply_random_policy(
#         tf_policies, image)
#     return augmented_image


# def distort_image_with_autoaugment(image, augmentation_name):
#     """Applies the AutoAugment policy to `image`.
#
#   AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
#
#   Args:
#     image: `Tensor` of shape [height, width, 3] representing an image.
#     augmentation_name: The name of the AutoAugment policy to use. The available
#       options are `v0` and `tests`. `v0` is the policy used for
#       all of the results in the paper and was found to achieve the best results
#       on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
#       found on the COCO dataset that have slight variation in what operations
#       were used during the search procedure along with how many operations are
#       applied in parallel to a single image (2 vs 3).
#
#   Returns:
#     A tuple containing the augmented versions of `image`.
#   """
#     available_policies = {'v0': policy_v0,
#                           'tests': policy_vtest}
#     if augmentation_name not in available_policies:
#         raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))
#
#     policy = available_policies[augmentation_name]()
#     # Hparams that will be used for AutoAugment.
#     augmentation_hparams = contrib_training.HParams(
#         cutout_const=100, translate_const=250)
#
#     return build_and_apply_nas_policy(policy, image, augmentation_hparams)


# @jax.jit

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

    image, geometric_transform = jax.lax.cond(
        selected_op == 0,
        lambda op: (color_transforms.autocontrast(op[0], *op[1][0]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 1,
        lambda op: (color_transforms.equalize(op[0], *op[1][1]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 2,
        lambda op: (color_transforms.invert(op[0], *op[1][2]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 3,
        lambda op: (color_transforms.posterize(op[0], *op[1][3]).astype('uint8'),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 4,
        lambda op: (color_transforms.solarize(op[0], *op[1][4]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 5,
        lambda op: (color_transforms.solarize_add(op[0], *op[1][5]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 6,
        lambda op: (color_transforms.color(op[0], *op[1][6]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 7,
        lambda op: (color_transforms.contrast(op[0], *op[1][7]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 8,
        lambda op: (color_transforms.brightness(op[0], *op[1][8]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 9,
        lambda op: (color_transforms.sharpness(op[0], *op[1][9]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 10,
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.rotate(*op[1][10]))),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 11,
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.shear(*op[1][11]))),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 12,
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.shear(*op[1][12]))),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 13,
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.translate(*op[1][13]))),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    image, geometric_transform = jax.lax.cond(
        selected_op == 14,
        lambda op: (op[0], jnp.matmul(geometric_transform,
                                      transforms.translate(*op[1][14]))),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    # currently not jittable
    image, geometric_transform = jax.lax.cond(
        selected_op == 15,
        lambda op: (color_transforms.cutout(op[0], *op[1][15]),
                    geometric_transform),
        lambda op: (op[0], geometric_transform),
        (image, args)
    )
    return image, geometric_transform


# @jax.jit
def _randaugment_inner_for_loop(_, in_args):
    """
    Loop body for for randougment.
    Args:
        _:
        in_args:

    Returns:

    """
    (image, geometric_transforms, random_key, available_ops, op_probs,
     magnitude, cutout_const, translate_const, join_transforms) = in_args
    random_keys = random.split(random_key, num=8)
    random_key = random_keys[0]  # keep for next iteration
    op_to_select = random.choice(random_keys[1], available_ops, p=op_probs)
    mask_value = random.randint(random_keys[2], [image.shape[-1]], minval=-1, maxval=256)
    random_magnitude = random.uniform(random_keys[3], [], minval=0., maxval=magnitude)
    if DEBUG:
        cutout_mask = color_transforms.get_random_cutout_mask(
            random_keys[4],
            image.shape,
            40)
    else:
        cutout_mask = jnp.zeros_like(image[:, :, :1]).astype('bool')

    translate_vals = (random.uniform(random_keys[5], [], minval=0.0, maxval=1.0) * translate_const,
                      random.uniform(random_keys[6], [], minval=0.0, maxval=1.0) * translate_const)
    negate = random.randint(random_keys[7], [], minval=0, maxval=2).astype('bool')

    args = level_to_arg(cutout_mask, translate_vals, negate, random_magnitude)

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
           magnitude, cutout_const, translate_const, join_transforms)


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
        cutout_const:
        translate_const:
        default_replace_value:
        available_ops:
        op_probs:
        join_transforms:

    Returns:
        The augmented version of `image`.
    """

    geometric_transforms = jnp.identity(4)

    for_i_args = (image, geometric_transforms, random_key, available_ops, op_probs,
                  magnitude, cutout_const, translate_const, join_transforms)

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
