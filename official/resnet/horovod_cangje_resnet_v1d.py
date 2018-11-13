from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

import tensorflow as tf
import numpy as np

import horovod.tensorflow as hvd
from official.resnet import imagenet_preprocessing_cangje
from official.resnet.horovod_estimator import HorovodEstimator, lp_debug, BroadcastGlobalVariablesHook, lp_debug_rank0, \
    AllReduceTensorHook, ConfusionMatrixHook, EvalImageVisualizationHook
from official.resnet.slim.nets import nets_factory, tf_resnet_v1d_model

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}
_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

tf.logging.set_verbosity(tf.logging.INFO)


def get_filenames(is_training, data_dir, test=False):
    if not test:
        if is_training:
            return [os.path.join(data_dir, i) for i in filter(lambda x: x.startswith('train'), os.listdir(data_dir))]
        else:
            return [os.path.join(data_dir, i) for i in filter(lambda x: x.startswith('val'), os.listdir(data_dir))]
    else:
        return [os.path.join(data_dir, i) for i in filter(lambda x: x.startswith('test'), os.listdir(data_dir))]


def _parse_example_proto(example_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
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
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = imagenet_preprocessing_cangje.preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        num_channels=_NUM_CHANNELS,
        is_training=is_training,
        resize_min=_RESIZE_MIN)
    image = tf.cast(image, dtype)

    return image, label


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float16):
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            # num_parallel_calls=1,
            drop_remainder=False))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None, dtype=tf.float16, num_shards=1, shard_index=0, test=False, seed=None):
    filenames = get_filenames(is_training, data_dir, test=test)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES, seed=seed)

    dataset = dataset.shard(num_shards, shard_index)

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        num_gpus=num_gpus,
        dtype=dtype
    )


def model_fn_label_smoothing(features, labels, mode, params):
    tf.summary.image('features ', features, max_outputs=8)
    
    raw_features = tf.identity(features, 'features')
    raw_labels =  tf.identity(labels, 'labels')
    classes = flags_obj.num_classes

    dtype = params['dtype']
    assert features.dtype == dtype

    if mode == tf.estimator.ModeKeys.TRAIN:
        if flags_obj.label_smoothing:
            lp_debug_rank0('using label smoothing')
            eta = 0.1
        else:
            eta = 0.0

        if mode == tf.estimator.ModeKeys.TRAIN:
            if flags_obj.mixup:
                lp_debug_rank0('using mixup')
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                features = lam * features + (1 - lam) * features[::-1]
                y1 = tf.one_hot(labels, classes, on_value=1-eta + eta/classes, off_value=eta/classes)
                y2 = tf.one_hot(labels[::-1], classes, on_value=1-eta + eta/classes, off_value=eta/classes)
                labels = lam * y1 + (1 - lam) * y2
            else:
                labels = tf.one_hot(labels, classes, on_value=1-eta + eta/classes, off_value=eta/classes)
    else:
        labels = tf.one_hot(labels, classes)

    # model = nets_factory.get_network_fn(flags_obj.model_type, classes, is_training=mode == tf.estimator.ModeKeys.TRAIN)

    model = tf_resnet_v1d_model.ImagenetModel(mode == tf.estimator.ModeKeys.TRAIN, 50, num_classes=classes, resnet_version=1, dtype=dtype, data_format=flags_obj.data_format)
    logits = model(features)

    logits = tf.cast(logits, tf.float32)
    predicts = tf.argmax(input=logits, axis=1)
    tf.identity(predicts, 'predicts')

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=1.0)
    aux_loss = tf.constant(0.0)

    def exclude_batch_norm_and_bias(name):
        return 'batch_normalization' not in name and 'BatchNorm' not in name and 'biases' not in name

    trainable_variables = tf.trainable_variables()
    trainable_variables_without_bn = [v for v in tf.trainable_variables() if exclude_batch_norm_and_bias(v.name)]
    global_variables = tf.global_variables()

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    lp_debug_rank0('global_variables {}'.format(len(global_variables)))
    lp_debug_rank0('trainable_variables {}'.format(len(trainable_variables)))
    lp_debug_rank0('trainable_variables_without_bn size {}'.format(len(trainable_variables_without_bn)))
    lp_debug_rank0('regularization_losses size {}'.format(len(regularization_losses)))

    l2_loss = flags_obj.weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                                 for v in trainable_variables_without_bn])
    loss = cross_entropy + aux_loss + l2_loss

    tf.identity(cross_entropy, name='cross_entropy')
    tf.identity(aux_loss, name='aux_loss')
    tf.identity(l2_loss, 'l2_loss')
    tf.identity(loss, name='loss')

    accuracy = tf.metrics.accuracy(raw_labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=raw_labels, k=5, name='top_5_op'))
    rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=tf.nn.softmax(logits))

    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.identity(rmse[1], name='rmse')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = params['learning_rate_fn'](global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        optimizer = hvd.DistributedOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            avg_grad_vars = optimizer.compute_gradients(loss)
            minimize_op = optimizer.apply_gradients(avg_grad_vars, global_step)

        train_op = tf.group(minimize_op, update_ops)

        tf.identity(learning_rate, name='learning_rate')
        lp_debug_rank0('update_ops size {}'.format(len(update_ops)))

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
    hvd.init()
    lp_debug_rank0('flag_obj: {}'.format(flags_obj))
    from tensorflow.python.framework import ops
    lp_debug('seed {} {}'.format(ops.get_default_graph().seed, ops.get_default_graph()._last_id))

    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=4,
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    batches_per_epoch = flags_obj.num_images // (flags_obj.batch_size * hvd.size())
    initial_lr = flags_obj.base_lr * (flags_obj.batch_size * hvd.size() / 256)

    def learning_rate_fn(global_step):
        warmup_steps = int(batches_per_epoch * 5)
        total_steps = flags_obj.train_epochs * batches_per_epoch
        lr = tf.train.cosine_decay(initial_lr, global_step, total_steps - warmup_steps)
        warmup_lr = initial_lr * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
        return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    DTYPE_MAP = {'fp16': tf.float16, 'fp32': tf.float32 }
    dtype = DTYPE_MAP[flags_obj.dtype]

    model_dir = flags_obj.model_dir if hvd.rank() == 0 else None
    classifier = HorovodEstimator(model_fn=model_fn_label_smoothing, model_dir=model_dir,
                                  config=tf.estimator.RunConfig(session_config=session_config, save_checkpoints_steps=flags_obj.save_checkpoints_steps),
                                  params={'learning_rate_fn': learning_rate_fn, 'dtype': dtype})

    def input_fn_train(num_epochs, dtype=tf.float16):
        return input_fn(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=num_epochs,
            dtype=dtype,
            num_shards=hvd.size(),
            shard_index=hvd.rank())

    def input_fn_eval(dtype=tf.float16):
        return input_fn(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=1,
            dtype=dtype)

    def input_fn_test(dtype=tf.float16):
        return input_fn(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=1,
            dtype=dtype,
            test=True)

    tensors_to_log = {"top1": 'train_accuracy', 'top5': 'train_accuracy_top_5', 'rmse': 'rmse', 'lr': 'learning_rate', 'loss': 'loss', 'l2_loss': 'l2_loss', 'cross_entropy': 'cross_entropy', 'aux_loss': 'aux_loss'}
    all_reduce_hook = AllReduceTensorHook(tensors_to_log, model_dir)
    init_hooks = BroadcastGlobalVariablesHook(0)
    init_restore_hooks = BroadcastGlobalVariablesHook(0, pretrained_model_path=flags_obj.pretrained_model_path, fine_tune=flags_obj.fine_tune, exclusions=nets_factory.exclusion_for_training[flags_obj.model_type])

    cm_hook = ConfusionMatrixHook(flags_obj.num_classes, 'features', 'labels', 'predicts', summary_dir=model_dir)
    visualization_hook = EvalImageVisualizationHook('features', 'labels', 'predicts', summary_dir=model_dir, every_n_steps=20)

    if flags_obj.evaluate:
        val_hooks = [init_hooks, visualization_hook]
        if flags_obj.confusion_matrix:
            val_hooks.append(cm_hook)

        if hvd.rank() == 0:
            lp_debug('begin evaluate')
            eval_results = classifier.evaluate(input_fn=lambda: input_fn_eval(dtype=dtype), hooks=val_hooks)
            lp_debug(eval_results)
            lp_debug('end evaluate')
        else:
            time.sleep(60 * 10)
        return

    if flags_obj.test:
        test_hooks = [init_hooks, visualization_hook]
        if flags_obj.confusion_matrix:
            test_hooks.append(cm_hook)

        if hvd.rank() == 0:
            lp_debug('begin test')
            eval_results = classifier.evaluate(input_fn=lambda: input_fn_test(dtype=dtype), hooks=test_hooks)
            lp_debug(eval_results)
            lp_debug('end test')
        else:
            time.sleep(60 * 10)
        return

    continue_train_epoch = flags_obj.continue_train_epoch
    rest_train_epoch = flags_obj.train_epochs - continue_train_epoch
    n_loops = math.ceil(rest_train_epoch / flags_obj.epochs_between_evals)
    schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
    schedule[-1] = rest_train_epoch - sum(schedule[:-1])  # over counting.
    schedule.insert(0, continue_train_epoch)

    for cycle_index, num_train_epochs in enumerate(schedule):
        lp_debug_rank0('Starting cycle: {}/{}'.format(cycle_index, int(n_loops)))

        if num_train_epochs:
            train_hooks = [all_reduce_hook]

            if cycle_index == 0:
                train_hooks.append(init_restore_hooks)
                lp_debug_rank0('will restore from {}'.format(flags_obj.pretrained_model_path))
            else:
                train_hooks.append(init_hooks)

            classifier.train(input_fn=lambda: input_fn_train(num_train_epochs, dtype=dtype),
                             hooks=train_hooks, max_steps=None)

            if hvd.rank() == 0:
                lp_debug('begin evaluate')
                val_hooks = []
                if flags_obj.confusion_matrix:
                    val_hooks.append(cm_hook)

                eval_results = classifier.evaluate(input_fn=lambda: input_fn_eval(dtype=dtype), hooks=val_hooks)
                lp_debug(eval_results)
                lp_debug('end evaluate')
            else:
                # should wait for rank0 to finish evaluating
                lp_debug('begin to sleep')
                time.sleep(90)
                lp_debug('end to sleep')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='', type=str, default='/home/liupeng/data/imagenet_tfrecord')
    parser.add_argument('--model_type', help='', type=str, default='resnet_v1_50')
    parser.add_argument('--model_dir', help='', type=str, default='model_dir')
    parser.add_argument('--batch_size', help='', type=int, default=256)
    parser.add_argument('--train_epochs', help='', type=int, default=200)
    parser.add_argument('--epochs_between_evals', help='', type=int, default=1)
    parser.add_argument('--continue_train_epoch', help='', type=int, default=1)
    parser.add_argument('--save_checkpoints_steps', help='', type=int, default=1200)
    parser.add_argument('--num_classes', help='', type=int, default=1001)
    parser.add_argument('--base_lr', help='', type=float, default=0.1)
    parser.add_argument('--resize_min', help='', type=int, default=256)
    parser.add_argument('--pretrained_model_path', help='', type=str)
    parser.add_argument('--evaluate', help='', action='store_true')
    parser.add_argument('--test', help='', action='store_true')
    parser.add_argument('--fine_tune', help='', action='store_true')
    parser.add_argument('--label_smoothing', help='', action='store_true')
    parser.add_argument('--mixup', help='', action='store_true')
    parser.add_argument('--confusion_matrix', help='', action='store_true')
    parser.add_argument('--num_images', help='', type=int, default=1281167)
    parser.add_argument('--weight_decay', help='', type=float, default=1e-4)
    parser.add_argument('--dtype', help='', type=str, default='fp32')
    parser.add_argument('--data_format', help='', type=str, default='channels_first')

    flags_obj = parser.parse_args()

    # 299
    _DEFAULT_IMAGE_SIZE = nets_factory.get_network_fn(flags_obj.model_type).default_image_size
    _RESIZE_MIN = flags_obj.resize_min
    _NUM_CHANNELS = 3

    tf.app.run()
