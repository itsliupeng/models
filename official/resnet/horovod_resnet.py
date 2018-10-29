#  Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

import horovod.tensorflow as hvd
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_run_loop
from official.resnet.horovod_estimator import HorovodEstimator, lp_debug, BroadcastGlobalVariablesHook, lp_debug_rank0, \
    AllReduceTensorHook

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

tf.logging.set_verbosity(tf.logging.INFO)

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


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None, dtype=tf.float32, num_shards=1, shard_index=0):
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
    dataset = tf.data.Dataset.from_tensor_slices(filenames).shard(num_shards, shard_index)

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


def cnn_model_fn(features, labels, mode, params):
    weight_decay = 1e-4
    momentum = 0.9

    from official.resnet.slim import resnet_model
    model = resnet_model.ResNet(resnet_size=50)
    logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.cast(logits, tf.float32)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, weights=1.0)
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    regularization_variables = [v for  v in tf.trainable_variables() if exclude_batch_norm(v.name)]

    lp_debug_rank0('REGULARIZATION_Varaibles: {}'.format(regularization_variables))

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in regularization_variables])
    loss = cross_entropy + l2_loss

    tf.identity(l2_loss, 'l2_loss')
    tf.identity(loss, name='loss')
    tf.identity(cross_entropy, name='cross_entropy')

    if hvd.rank() == 0:
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('l2_loss', l2_loss)

        lp_debug_rank0('REGULARIZATION_LOSSES: {}'.format(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))

    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = params['learning_rate_fn'](global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        optimizer = hvd.DistributedOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            # lp: do reduce
            avg_grad_vars = optimizer.compute_gradients(loss)
            minimize_op = optimizer.apply_gradients(avg_grad_vars, global_step)


        train_op = tf.group(minimize_op, update_ops)

        tf.identity(learning_rate, name='learning_rate')

        if hvd.rank() == 0:
            # Create a tensor named learning_rate for logging purposes
            tf.summary.scalar('learning_rate', learning_rate)
            # Create a tensor named train_accuracy for logging purposes
            tf.summary.scalar('train_accuracy', accuracy[1])
            tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

            lp_debug_rank0('update_ops : {}'.format(update_ops))

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    else:
        if hvd.rank() == 0:
            tf.identity(accuracy[1], name='eval_accuracy')
            tf.identity(accuracy_top_5[1], name='eval_accuracy_top_5')
            tf.summary.scalar('eval_accuracy', accuracy[1])
            tf.summary.scalar('eval_accuracy_top_5', accuracy_top_5[1])

        metrics = {'val_accuracy': accuracy, 'val_accuracy_top_5': accuracy_top_5}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=metrics)


def main(unused_argv):
    # Horovod: initialize Horovod.
    hvd.init()

    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=4,
        allow_soft_placement=True)


    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=flags_obj.batch_size * hvd.size(), batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=True, base_lr=.128*0.6)

    model_dir = './mnist_convnet_model' if hvd.rank() == 0 else None

    # Create the tf.estimator
    # classifier = HorovodEstimator(model_fn=cnn_model_fn, model_dir=model_dir,
    #                               config=tf.estimator.RunConfig(session_config=session_config, save_checkpoints_steps=flags_obj.save_checkpoints_steps),
    #                               params={'learning_rate_fn': learning_rate_fn})

    def input_fn_train(num_epochs):
        return input_fn(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=num_epochs, num_shards=hvd.size(), shard_index=hvd.rank())

    def input_fn_eval():
        return input_fn(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=1, num_shards=hvd.size(), shard_index=hvd.rank())

    tensors_to_log = {"top1": 'train_accuracy', 'top5': 'train_accuracy_top_5', 'lr': 'learning_rate', 'loss': 'loss', 'l2_loss': 'l2_loss', 'cross_entropy': 'cross_entropy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    all_reduce_hook = AllReduceTensorHook(tensors_to_log, model_dir, every_n_iter=100)
    init_hooks = BroadcastGlobalVariablesHook(0)

    # if flags_obj.evaluate:
    #     lp_debug('begin evaluate')
    #     eval_results = classifier.evaluate(input_fn=input_fn_eval, hooks=[init_hooks])
    #     lp_debug(eval_results)
    #     lp_debug('end evaluate')
    #     return


    n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
    schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
    schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.

    for cycle_index, num_train_epochs in enumerate(schedule):
        lp_debug('Starting cycle: {}/{}'.format(cycle_index, int(n_loops)))

        if num_train_epochs:

            # Create the tf.estimator
            classifier = HorovodEstimator(model_fn=cnn_model_fn, model_dir=model_dir,
                                          config=tf.estimator.RunConfig(session_config=session_config,
                                                                        save_checkpoints_steps=flags_obj.save_checkpoints_steps),
                                          params={'learning_rate_fn': learning_rate_fn})

            if hvd.rank() == 0:
                train_hooks = [logging_hook, all_reduce_hook]
            else:
                train_hooks = [all_reduce_hook]

            train_hooks.append(init_hooks)

            classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                             hooks=train_hooks, steps=200, max_steps=None)

            lp_debug('begin evaluate')
            eval_results = classifier.evaluate(input_fn=input_fn_eval, hooks=[init_hooks])
            lp_debug(eval_results)
            lp_debug('end evaluate')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='', type=str, default='/home/liupeng/data/imagenet_tfrecord')
    parser.add_argument('--batch_size', help='', type=int, default=32)
    parser.add_argument('--train_epochs', help='', type=int, default=90)
    parser.add_argument('--epochs_between_evals', help='', type=int, default=90)
    parser.add_argument('--save_checkpoints_steps', help='', type=int, default=600)
    parser.add_argument('--evaluate', help='', action='store_true')

    flags_obj = parser.parse_args()

    tf.app.run()
