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

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

NAME_TO_FUNC = {
    # returns tuple of (function, is_geometric_transform: bool)
    'AutoContrast':  (color_transforms.autocontrast, False),
    'Equalize': (color_transforms.equalize, False),
    'Invert': (color_transforms.invert, False),
    'Posterize': (color_transforms.posterize, False),
    'Solarize': (color_transforms.solarize, False),
    'SolarizeAdd': (color_transforms.solarize_add, False),
    'Color': (color_transforms.color, False),
    'Contrast': (color_transforms.contrast, False),
    'Brightness': (color_transforms.brightness, False),
    'Sharpness': (color_transforms.sharpness, False),
    'Cutout': (color_transforms.cutout, False),
    'Rotate': (transforms.rotate, True),
    'ShearX': (transforms.shear, True),
    'ShearY': (transforms.shear, True),
    'TranslateX': (transforms.translate, True),
    'TranslateY': (transforms.translate, True),
}

AVAILABLE_OPS = tuple(NAME_TO_FUNC.keys())


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return 1.0  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return level


def _enhance_level_to_arg(level):
    return (level / _MAX_LEVEL) * 1.8 + 0.1,


def _rotate_level_to_arg(level, negate):
    level = (level / _MAX_LEVEL)
    if negate:
        return -level
    return level


def _shear_level_to_arg(level, negate):
    level = (level / _MAX_LEVEL)
    # Flip level to negative with 50% chance.
    if negate:
        return -level
    return level


def _translate_level_to_arg(translate_val, negate):
    # Flip level to negative with 50% chance.
    if negate:
        return -translate_val[0], -translate_val[1]
    return translate_val


def level_to_arg(cutout_val, translate_val, negate):
    """
    Translates the level to args for various functions.
    Args:
        cutout_val: value for cutout size of cutout function
        translate_val: value for
        negate:

    Returns:

    """

    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Posterize': lambda level: (5 - min(4, int(level / _MAX_LEVEL * 4)),),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'Cutout': lambda level: (cutout_val,),
        'Rotate': lambda level: (_rotate_level_to_arg(level, negate),),
        'ShearX': lambda level: (_shear_level_to_arg(level, negate), 0),
        'ShearY': lambda level: (0, _shear_level_to_arg(level, negate)),
        'TranslateX': lambda level: (_translate_level_to_arg(translate_val, negate)[0], 0.),
        'TranslateY': lambda level: (0., _translate_level_to_arg(translate_val, negate)[1]),
    }


def _parse_policy_info(name, prob, level, cutout_val, translate_val, negate):
    """Return the function that corresponds to `name` and update `level` param."""
    func, is_geometric_transform = NAME_TO_FUNC[name]
    args = level_to_arg(cutout_val, translate_val, negate)[name](level)
    return func, is_geometric_transform, prob, args


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


def distort_image_with_randaugment(image,
                                   num_layers,
                                   magnitude,
                                   random_key,
                                   cutout_const=40,
                                   translate_const=50.0,
                                   default_replace_value=None,
                                   available_ops=AVAILABLE_OPS,
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
        join_transforms:

    Returns:
        The augmented version of `image`.
    """

    geometric_transforms = []
    print('num:', num_layers)
    print()
    for layer_num in range(num_layers):
        random_keys = random.split(random_key, num=8)
        random_key = random_keys[0]  # keep for next iteration
        op_to_select = random.randint(random_keys[1], [], minval=0, maxval=len(available_ops)).item()
        print('chosen:', available_ops[op_to_select])
        op_name = available_ops[op_to_select]
        prob = random.uniform(random_keys[2], [], minval=0.2, maxval=0.8).item()
        random_magnitude = random.uniform(random_keys[3], [], minval=0., maxval=magnitude).item()
        cutout_mask = color_transforms.get_random_cutout_mask(
            random_keys[4],
            image.shape,
            max_mask_shape=(cutout_const, cutout_const))
        translate_vals = (random.uniform(random_keys[5], [], minval=0.0, maxval=1.0).item() * translate_const,
                          random.uniform(random_keys[6], [], minval=0.0, maxval=1.0).item() * translate_const)
        negate = random.randint(random_keys[7], [], minval=0, maxval=2).astype('bool').item()
        func, is_geometric_transform, _, args = _parse_policy_info(op_name,
                                                                   prob,              # random
                                                                   random_magnitude,  # random
                                                                   cutout_mask,       # random
                                                                   translate_vals,    # random
                                                                   negate)            # random
        if is_geometric_transform:
            geometric_transforms.append(func(*args))
        else:
            image = func(image, *args)

    if len(geometric_transforms) > 0:
        if join_transforms:
            replace_value = default_replace_value or random.randint(random_key,
                                                                    [image.shape[-1]],
                                                                    minval=0,
                                                                    maxval=256)
            final_transform = jnp.identity(4)
            for t in geometric_transforms:
                final_transform = jnp.matmul(final_transform, t)

            image = transforms.apply_transform(image, final_transform, mask_value=replace_value)
        else:
            for t in geometric_transforms:
                random_key, split_key = random.split(random_key)
                replace_value = default_replace_value or random.randint(split_key,
                                                                        [image.shape[-1]],
                                                                        minval=0,
                                                                        maxval=256)
                curr_transform = t
                image = transforms.apply_transform(image, curr_transform, mask_value=replace_value)
    return image
