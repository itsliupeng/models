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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags

from official.resnet import imagenet_preprocessing
from official.resnet import resnet_model
from official.resnet import resnet_run_loop
from official.utils.flags import core as flags_core
from official.utils.logs import logger

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % i)
            for i in range(128)]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
  
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):
  
      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>
  
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
  
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
    """Parses a record containing a training example of an image.
  
    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).
  
    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.
  
    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training)
    image = tf.cast(image, dtype)

    return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32):
    """Input function which provides batches for train or eval.
  
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      num_gpus: The number of gpus used for training.
      dtype: Data type to use for images/features
  
    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))

    return resnet_run_loop.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        num_gpus=num_gpus,
        examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
        dtype=dtype
    )


def get_synth_input_fn(dtype):
    return resnet_run_loop.get_synth_input_fn(
        _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES,
        dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for Imagenet data.
    
        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          resnet_version: Integer representing which version of the ResNet network
            to use. See README for details. Valid values: [1, 2]
          dtype: The TensorFlow dtype to use for calculations.
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        super(ImagenetModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
  
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
  
    Args:
      resnet_size: The number of convolutional layers needed in the model.
  
    Returns:
      A list of block sizes to use in building the model.
  
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
        raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""

    # Warmup and higher lr may not be valid for fine tuning with small batches
    # and smaller numbers of training images.
    if params['fine_tune']:
        warmup = False
        base_lr = .1
    else:
        warmup = True
        base_lr = .128

    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=warmup, base_lr=base_lr)


    # return resnet_run_loop.resnet_model_fn(
    #     features=features,
    #     labels=labels,
    #     mode=mode,
    #     model_class=ImagenetModel,
    #     resnet_size=params['resnet_size'],
    #     weight_decay=1e-4,
    #     learning_rate_fn=learning_rate_fn,
    #     momentum=0.9,
    #     data_format=params['data_format'],
    #     resnet_version=params['resnet_version'],
    #     loss_scale=params['loss_scale'],
    #     loss_filter_fn=None,
    #     dtype=params['dtype'],
    #     fine_tune=params['fine_tune']
    # )

    resnet_size = params['resnet_size']
    weight_decay = 1e-4
    learning_rate_fn = learning_rate_fn
    momentum = 0.9
    data_format = params['data_format']
    resnet_version = params['resnet_version']
    loss_filter_fn = None
    dtype = params['dtype']

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == dtype

    model = ImagenetModel(resnet_size, data_format, resnet_version=resnet_version,
                        dtype=dtype)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })


    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        grad_vars = optimizer.compute_gradients(loss)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                    targets=labels,
                                                    k=5,
                                                    name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)



def define_imagenet_flags():
    resnet_run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'])
    flags.adopt_module_key_flags(resnet_run_loop)
    flags_core.set_defaults(train_epochs=90)


def run_imagenet(flags_obj):
    """Run ResNet ImageNet training and eval loop.
  
    Args:
      flags_obj: An object containing parsed flag values.
    """
    input_function = (flags_obj.use_synthetic_data and
                      get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                      input_fn)

    resnet_run_loop.resnet_main(
        flags_obj, imagenet_model_fn, input_function, DATASET_NAME,
        shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
    with logger.benchmark_context(flags.FLAGS):
        run_imagenet(flags.FLAGS)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_imagenet_flags()
    absl_app.run(main)
