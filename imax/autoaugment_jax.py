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
import inspect
from jax import random
import jax.numpy as jnp
import color_transforms
import transforms

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def policy_v0():
    """Autoaugment policy that was used in AutoAugment Paper."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    return policy


def policy_vtest():
    """Autoaugment test policy for debugging."""
    # Each tuple is an augmentation operation of the form
    # (operation, probability, magnitude). Each element in policy is a
    # sub-policy that will be applied sequentially on the image.
    policy = [
        [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
    ]
    return policy


NAME_TO_FUNC = {
    'AutoContrast':  color_transforms.autocontrast,
    'Equalize': color_transforms.equalize,
    'Invert': color_transforms.invert,
    'Posterize': color_transforms.posterize,
    'Solarize': color_transforms.solarize,
    'SolarizeAdd': color_transforms.solarize_add,
    'Color': color_transforms.color,
    'Contrast': color_transforms.contrast,
    'Brightness': color_transforms.brightness,
    'Sharpness': color_transforms.sharpness,
    'Rotate': transforms.rotate,
    'ShearX': transforms.shear_x,
    'ShearY': transforms.shear_y,
    'TranslateX': transforms.translate_x,
    'TranslateY': transforms.translate_y,
    'Cutout': transforms.cutout,
}


def _randomly_negate_tensor(tensor, random_key):
    """With 50% prob turn the tensor negative."""
    random_key, subkey = random.split(random_key)
    should_flip = jnp.floor(random.uniform(subkey, shape=()) + 0.5).astype('bool')
    if should_flip:
        return tensor
    return -tensor


def _rotate_level_to_arg(level, random_key):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level, random_key)
    return level


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return 1.0  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return level


def _enhance_level_to_arg(level):
    return (level / _MAX_LEVEL) * 1.8 + 0.1


def _shear_level_to_arg(level, random_key):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, random_key)
    return level


def _translate_level_to_arg(level, translate_const, random_key):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, random_key)
    return level


def level_to_arg(hparams, random_key):
    # TODO split random key and pass it to funcs
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': lambda level: (int((level / _MAX_LEVEL) * hparams.cutout_const),),
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level_to_arg(
            level, hparams.translate_const),
        'TranslateY': lambda level: _translate_level_to_arg(
            level, hparams.translate_const),
        # pylint:enable=g-long-lambda
    }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams)[name](level)

    # Check to see if prob is passed into function. This is used for operations
    # where we alter bboxes independently.
    # pytype:disable=wrong-arg-types
    if 'prob' in inspect.getargspec(func)[0]:
        args = tuple([prob] + list(args))
    # pytype:enable=wrong-arg-types

    # Add in replace arg if it is required for the function that is being called.
    # pytype:disable=wrong-arg-types
    if 'replace' in inspect.getargspec(func)[0]:
        # Make sure replace is the final argument
        assert 'replace' == inspect.getargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])
    # pytype:enable=wrong-arg-types

    return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)

    # If prob is a function argument, then this randomness is being handled
    # inside the function, so make sure it is always called.
    # pytype:disable=wrong-arg-types
    if 'prob' in inspect.getargspec(func)[0]:
        prob = 1.0
    # pytype:enable=wrong-arg-types

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(
        tf.floor(tf.random_uniform([], dtype=tf.float32) + prob), tf.bool)
    augmented_image = tf.cond(
        should_apply_op,
        lambda: func(image, *args),
        lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = tf.random_uniform([], maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(policies):
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda selected_policy=policy: selected_policy(image),
            lambda: image)
    return image


def build_and_apply_nas_policy(policies, image,
                               augmentation_hparams):
    """Build a policy from the given policies passed in and apply to image.

  Args:
    policies: list of lists of tuples in the form `(func, prob, level)`, `func`
      is a string name of the augmentation function, `prob` is the probability
      of applying the `func` operation, `level` is the input argument for
      `func`.
    image: tf.Tensor that the resulting policy will be applied to.
    augmentation_hparams: Hparams associated with the NAS learned policy.

  Returns:
    A version of image that now has data augmentation applied to it based on
    the `policies` pass into the function.
  """
    replace_value = [128, 128, 128]

    # func is the string name of the augmentation function, prob is the
    # probability of applying the operation and level is the parameter associated
    # with the tf op.

    # tf_policies are functions that take in an image and return an augmented
    # image.
    tf_policies = []
    for policy in policies:
        tf_policy = []
        # Link string name to the correct python function and make sure the correct
        # argument is passed into that function.
        for policy_info in policy:
            policy_info = list(policy_info) + [replace_value, augmentation_hparams]

            tf_policy.append(_parse_policy_info(*policy_info))

        # Now build the tf policy that will apply the augmentation procedue
        # on image.
        def make_final_policy(tf_policy_):
            def final_policy(image_):
                for func, prob, args in tf_policy_:
                    image_ = _apply_func_with_prob(
                        func, image_, args, prob)
                return image_

            return final_policy

        tf_policies.append(make_final_policy(tf_policy))

    augmented_image = select_and_apply_random_policy(
        tf_policies, image)
    return augmented_image


def distort_image_with_autoaugment(image, augmentation_name):
    """Applies the AutoAugment policy to `image`.

  AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    augmentation_name: The name of the AutoAugment policy to use. The available
      options are `v0` and `test`. `v0` is the policy used for
      all of the results in the paper and was found to achieve the best results
      on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
      found on the COCO dataset that have slight variation in what operations
      were used during the search procedure along with how many operations are
      applied in parallel to a single image (2 vs 3).

  Returns:
    A tuple containing the augmented versions of `image`.
  """
    available_policies = {'v0': policy_v0,
                          'test': policy_vtest}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

    policy = available_policies[augmentation_name]()
    # Hparams that will be used for AutoAugment.
    augmentation_hparams = contrib_training.HParams(
        cutout_const=100, translate_const=250)

    return build_and_apply_nas_policy(policy, image, augmentation_hparams)


def distort_image_with_randaugment(image, num_layers, magnitude):
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

  Returns:
    The augmented version of `image`.
  """
    replace_value = [128] * 3
    tf.logging.info('Using RandAug.')
    augmentation_hparams = contrib_training.HParams(
        cutout_const=40, translate_const=100)
    available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

    for layer_num in range(num_layers):
        op_to_select = tf.random_uniform(
            [], maxval=len(available_ops), dtype=tf.int32)
        random_magnitude = float(magnitude)
        with tf.name_scope('randaug_layer_{}'.format(layer_num)):
            for (i, op_name) in enumerate(available_ops):
                prob = tf.random_uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
                func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                                   replace_value, augmentation_hparams)
                image = tf.cond(
                    tf.equal(i, op_to_select),
                    # pylint:disable=g-long-lambda
                    lambda selected_func=func, selected_args=args: selected_func(
                        image, *selected_args),
                    # pylint:enable=g-long-lambda
                    lambda: image)
    return image
