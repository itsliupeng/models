# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256


def _crop_and_flip(image, bbox):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    cropped = tf.slice(image, bbox_begin, bbox_size)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.
  
    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
  
    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.
  
    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)
  
    Note that the rank of `image` must be known.
  
    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
      num_channels: number of color channels in the image that will be distorted.
  
    Returns:
      the centered image.
  
    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
  
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
  
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
  
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
  
    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
  
    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.
  
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
  
    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.
  
    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.image.resize_images(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)
  ])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather than adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, is_training=False, resize_min=_RESIZE_MIN):
    """Preprocesses the given image.
  
    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.
  
    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      num_channels: Integer depth of the image buffer for decoding.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
  
    Returns:
      A preprocessed image.
    """

    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    if is_training:
        image = _crop_and_flip(image, bbox)
        image = _resize_image(image, output_height, output_width)
        image = apply_with_random_selector(image,
            lambda x, ordering: distort_color(x, ordering, fast_mode=False),
            num_cases=4)
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        image = _aspect_preserving_resize(image, resize_min)
        image = _central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
