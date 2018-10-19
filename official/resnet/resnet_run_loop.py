# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# pylint: disable=g-bad-import-order
from absl import flags

from official.resnet import imagenet_preprocessing
from official.utils.flags import core as flags_core


# pylint: enable=g-bad-import-order


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float32):
    """Given a Dataset with raw records, return an iterator over the records.
  
    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      num_gpus: The number of gpus used for training.
      examples_per_epoch: The number of examples in an epoch.
      dtype: Data type to use for images/features.
  
    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """

    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_calls=1,
            drop_remainder=False))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
    """Returns an input function that returns a dataset with random data.
  
    This input_fn returns a data set that iterates over a set of random data and
    bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
    copy is still included. This used to find the upper throughput bound when
    tunning the full input pipeline.
  
    Args:
      height: Integer height that will be used to create a fake image tensor.
      width: Integer width that will be used to create a fake image tensor.
      num_channels: Integer depth that will be used to create a fake image tensor.
      num_classes: Number of classes that should be represented in the fake labels
        tensor
      dtype: Data type for features/images.
  
    Returns:
      An input_fn that can be used in place of a real one to return a dataset
      that can be used for iteration.
    """

    # pylint: disable=unused-argument
    def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
        """Returns dataset filled with random data."""
        # Synthetic input should be within [0, 255].
        inputs = tf.truncated_normal(
            [batch_size] + [height, width, num_channels],
            dtype=dtype,
            mean=127,
            stddev=60,
            name='synthetic_inputs')

        labels = tf.random_uniform(
            [batch_size],
            minval=0,
            maxval=num_classes - 1,
            dtype=tf.int32,
            name='synthetic_labels')
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
        data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return data

    return input_fn


def image_bytes_serving_input_fn(image_shape, dtype=tf.float32):
    """Serving input fn for raw jpeg images."""

    def _preprocess_image(image_bytes):
        """Preprocess a single raw image."""
        # Bounding box around the whole image.
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
        height, width, num_channels = image_shape
        image = imagenet_preprocessing.preprocess_image(
            image_bytes, bbox, height, width, num_channels, is_training=False)
        return image

    image_bytes_list = tf.placeholder(
        shape=[None], dtype=tf.string, name='input_tensor')
    images = tf.map_fn(
        _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
    return tf.estimator.export.TensorServingInputReceiver(
        images, {'image_bytes': image_bytes_list})


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
        base_lr=0.1, warmup=False):
    """Get a learning rate that decays step-wise as training progresses.
  
    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.
      base_lr: Initial learning rate scaled based on batch_denom.
      warmup: Run a 5 epoch warmup to the initial lr.
    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                    initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn


def define_resnet_flags(resnet_size_choices=None):
    """Add flags and validators for ResNet."""
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_enum(
        name='resnet_version', short_name='rv', default='1',
        enum_values=['1', '2'],
        help=flags_core.help_wrap(
            'Version of ResNet. (1 or 2) See README.md for details.'))
    flags.DEFINE_bool(
        name='fine_tune', short_name='ft', default=False,
        help=flags_core.help_wrap(
            'If True do not train any parameters except for the final layer.'))
    flags.DEFINE_string(
        name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
        help=flags_core.help_wrap(
            'If not None initialize all the network except the final layer with '
            'these values'))
    flags.DEFINE_boolean(
        name='eval_only', default=False,
        help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                  'the latest checkpoint.'))
    flags.DEFINE_boolean(
        name="image_bytes_as_serving_input", default=False,
        help=flags_core.help_wrap(
            'If True exports savedmodel with serving signature that accepts '
            'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
            'represents the image. The former is easier to use for serving at '
            'the expense of image resize/cropping being done as part of model '
            'inference. Note, this flag only applies to ImageNet and cannot '
            'be used for CIFAR.'))

    choice_kwargs = dict(
        name='resnet_size', short_name='rs', default='50',
        help=flags_core.help_wrap('The size of the ResNet model to use.'))

    if resnet_size_choices is None:
        flags.DEFINE_string(**choice_kwargs)
    else:
        flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
