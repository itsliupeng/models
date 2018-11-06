from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import estimator
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.eager import context
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.training.monitored_session import USE_DEFAULT, Scaffold, MonitoredSession, ChiefSessionCreator
from tensorflow.python.training.session_run_hook import SessionRunArgs

import horovod.tensorflow as hvd

mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import itertools
from PIL import Image

estimator.Estimator._assert_members_are_not_overridden = lambda _: None


def is_rank0():
    return hvd.rank() == 0


def lp_debug(msg):
    head = 'lp-debug rank{}/{} in {}: '.format(hvd.rank(), hvd.size(), socket.gethostname())
    tf.logging.info('{}: {}'.format(head, msg))


def lp_debug_rank0(msg):
    if is_rank0():
        head = 'lp-debug only rank{}/{} in {}: '.format(hvd.rank(), hvd.size(), socket.gethostname())
        tf.logging.info('{}: {}'.format(head, msg))


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, pretrained_model_path=None, exclusions=[], device=''):
        """Construct a new BroadcastGlobalVariablesHook that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
          root_rank:
            Rank that will send data, other ranks will receive data.
          device:
            Device to be used for broadcasting. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device
        self._pretrained_model_path = pretrained_model_path
        self._saver = None
        self._exclusions = exclusions

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = hvd.broadcast_global_variables(self.root_rank)

        if self._pretrained_model_path and hvd.rank() == 0:
            # exclusions = nets_factory.exclusion_for_training['inception_v3']

            variables_to_restore = []
            for var in tf.model_variables():
                excluded = False
                for exclusion in self._exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)

            lp_debug('model_variables len {}, restore len {}'.format(len(tf.model_variables()), len(variables_to_restore)))
            self._saver = tf.train.Saver(var_list=variables_to_restore)

    def after_create_session(self, session, coord):
        if self._saver:
            lp_debug_rank0('begin to restore from {}'.format(self._pretrained_model_path))
            self._saver.restore(session, self._pretrained_model_path)
            lp_debug_rank0('end to restore from {}'.format(self._pretrained_model_path))


        lp_debug_rank0('br begin after_create_session ')
        session.run(self.bcast_op)
        lp_debug_rank0('br end after_create_session')


class BroadcastBatchNormHook(tf.train.SessionRunHook):
    def __init__(self, root_rank, device=''):
        super(BroadcastBatchNormHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                # lp-todo: all_reduce first
                self.bcast_op = tf.group(*[tf.assign(var, hvd.broadcast(var, self.root_rank))
                                           for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS)])

    def after_create_session(self, session, coord):
        lp_debug('br begin')
        session.run(self.bcast_op)
        lp_debug('br end')


class AllReduceTensorHook(session_run_hook.SessionRunHook):
    def __init__(self, named_tensor, summary_dir=None, every_n_iter=100, print_rank0=True):
        self._named_tensor = named_tensor
        self._every_n_iter = every_n_iter
        self._print_rank0 = print_rank0
        self._summary_dir = summary_dir

    def begin(self):
        self.avg_ops = {'avg/{}'.format(tag): hvd.allreduce(basic_session_run_hooks._as_graph_element(tensor))
                                 for (tag, tensor) in self._named_tensor.items()}

        self._global_step_tensor = training_util._get_or_create_global_step_read()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)

        stats = []
        for tag, tensor in tensor_values.items():
            stats.append("%s = %s" % (tag, tensor))

        if self._print_rank0:
            if hvd.rank() == 0:
                logging.info("allreduce tensor: %s", ", ".join(stats))
        else:
            logging.info("allreduce tensor: %s", ", ".join(stats))

        np.set_printoptions(**original)

    def _summary(self, tensor_values, step):
        if self._summary_dir:
            writer = tf.summary.FileWriterCache.get(self._summary_dir)
            this_summary = tf.Summary()
            for tag, value in tensor_values.items():
                this_summary.value.add(tag=tag, simple_value=value)
                writer.add_summary(this_summary, step)

            writer.flush()

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step % self._every_n_iter == 0:
            avg_values = run_context.session.run(self.avg_ops)
            avg_values['step'] = global_step
            self._log_tensors(avg_values)
            self._summary(avg_values, global_step)


class VisualizationHook(basic_session_run_hooks.StepCounterHook):
    def __init__(self, features_name, labels_name, predicts_name, every_n_steps=100, every_n_secs=None, summary_dir=None):
        super(VisualizationHook, self).__init__(every_n_steps, every_n_secs, summary_dir)
        self._features_name = features_name
        self._labels_name = labels_name
        self._predicts_name = predicts_name

    def before_run(self, run_context):
        return SessionRunArgs({'features': basic_session_run_hooks._as_graph_element(self._features_name),
                               'labels': basic_session_run_hooks._as_graph_element(self._labels_name),
                               'predicts': basic_session_run_hooks._as_graph_element(self._predicts_name),
                               'global_step': self._global_step_tensor})


    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            if self._total_batch_size:
                image_tag = 'images_sec'
                image_count = float(steps_per_sec) * self._total_batch_size
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec),
                                         Summary.Value(tag=image_tag, simple_value=image_count)])
                logging.info("%s: %g, %s: %g, step: %g", self._summary_tag, steps_per_sec, image_tag, image_count, global_step)
            else:
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
                logging.info("%s: %g, step: %g", self._summary_tag, steps_per_sec, global_step)

            self._summary_writer.add_summary(summary, global_step)

    def after_run(self, run_context, run_values):
        _ = run_context

        stale_global_step = run_values.results['global_step']
        features = run_values.results['features']
        labels = run_values.results['labels']

        self._total_batch_size = features.shape[0] * hvd.size()

        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                    global_step)
                if elapsed_time is not None:
                    self._log_and_record(elapsed_steps, elapsed_time, global_step)

        # Check whether the global step has been increased. Here, we do not use the
        # timer.last_triggered_step as the timer might record a different global
        # step value such that the comparison could be unreliable. For simplicity,
        # we just compare the stale_global_step with previously recorded version.
        if stale_global_step == self._last_global_step:
            # Here, we use a counter to count how many times we have observed that the
            # global step has not been increased. For some Optimizers, the global step
            # is not increased each time by design. For example, SyncReplicaOptimizer
            # doesn't increase the global step in worker's main train step.
            self._global_step_check_count += 1
            if self._global_step_check_count % 20 == 0:
                self._global_step_check_count = 0
                logging.warning(
                    "It seems that global step (tf.train.get_global_step) has not "
                    "been increased. Current value (could be stable): %s vs previous "
                    "value: %s. You could increase the global step by passing "
                    "tf.train.get_global_step() to Optimizer.apply_gradients or "
                    "Optimizer.minimize.", stale_global_step, self._last_global_step)
        else:
            # Whenever we observe the increment, reset the counter.
            self._global_step_check_count = 0

        self._last_global_step = stale_global_step


class ImageCounterHook(basic_session_run_hooks.StepCounterHook):
    def __init__(self, features, labels, every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None):
        super(ImageCounterHook, self).__init__(every_n_steps, every_n_secs, output_dir, summary_writer)
        self._features = features
        self._labels = labels
        self._steps_per_run = 1

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs({'global_step': self._global_step_tensor, 'features': self._features, 'labels': self._labels})

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            if self._total_batch_size:
                image_tag = 'images_sec'
                image_count = float(steps_per_sec) * self._total_batch_size
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec),
                                         Summary.Value(tag=image_tag, simple_value=image_count)])
                logging.info("%s: %g, %s: %g, step: %g", self._summary_tag, steps_per_sec, image_tag, image_count, global_step)
            else:
                summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
                logging.info("%s: %g, step: %g", self._summary_tag, steps_per_sec, global_step)

            self._summary_writer.add_summary(summary, global_step)

    def after_run(self, run_context, run_values):
        _ = run_context

        stale_global_step = run_values.results['global_step']
        features = run_values.results['features']
        labels = run_values.results['labels']

        self._total_batch_size = features.shape[0] * hvd.size()

        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                    global_step)
                if elapsed_time is not None:
                    self._log_and_record(elapsed_steps, elapsed_time, global_step)

        # Check whether the global step has been increased. Here, we do not use the
        # timer.last_triggered_step as the timer might record a different global
        # step value such that the comparison could be unreliable. For simplicity,
        # we just compare the stale_global_step with previously recorded version.
        if stale_global_step == self._last_global_step:
            # Here, we use a counter to count how many times we have observed that the
            # global step has not been increased. For some Optimizers, the global step
            # is not increased each time by design. For example, SyncReplicaOptimizer
            # doesn't increase the global step in worker's main train step.
            self._global_step_check_count += 1
            if self._global_step_check_count % 20 == 0:
                self._global_step_check_count = 0
                logging.warning(
                    "It seems that global step (tf.train.get_global_step) has not "
                    "been increased. Current value (could be stable): %s vs previous "
                    "value: %s. You could increase the global step by passing "
                    "tf.train.get_global_step() to Optimizer.apply_gradients or "
                    "Optimizer.minimize.", stale_global_step, self._last_global_step)
        else:
            # Whenever we observe the increment, reset the counter.
            self._global_step_check_count = 0

        self._last_global_step = stale_global_step


class ConfusionMatrixHook(basic_session_run_hooks.SecondOrStepTimer, tf.train.SessionRunHook):
    def __init__(self, num_classes, features_name, labels_name, predicts_name, summary_dir):
        super(ConfusionMatrixHook, self).__init__(every_steps=1)
        self._num_classes = num_classes
        self._features_name = features_name
        self._labels_name = labels_name
        self._predicts_name = predicts_name
        self._all_labels = []
        self._all_predicts = []
        self._summary_dir = summary_dir

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()

    def before_run(self, run_context):
        return SessionRunArgs({'features': basic_session_run_hooks._as_graph_element(self._features_name),
                               'labels': basic_session_run_hooks._as_graph_element(self._labels_name),
                               'predicts': basic_session_run_hooks._as_graph_element(self._predicts_name),
                               'global_step': self._global_step_tensor})

    def after_run(self, run_context, run_values):
        _ = run_context

        features = run_values.results['features']
        labels = run_values.results['labels']
        predicts = run_values.results['predicts']

        self._global_step  = run_values.results['global_step']
        self._all_labels.append(labels)
        self._all_predicts.append(predicts)

    def make_image(self, tensor):
        """Convert an numpy representation image to Image protobuf"""
        from PIL import Image
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return Summary.Image(height=height,
                             width=width,
                             colorspace=channel,
                             encoded_image_string=image_string)

    def confusion_matrix_summary(self, tag, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.close('all')

        f_size = max(5, int(0.6 * len(classes)))
        plt.figure(figsize=(f_size, f_size))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        image = Image.open(buf).convert('RGB')
        tensor = np.asarray(image, dtype=np.uint8)
        image = self.make_image(tensor)

        return Summary(value=[Summary.Value(tag=tag, image=image)])

    def end(self, session):
        summary_writer = tf.summary.FileWriterCache.get(self._summary_dir)
        all_labels_np = np.concatenate(self._all_labels)
        all_predicts_np = np.concatenate(self._all_predicts)
        cnf_matrix = confusion_matrix(all_labels_np, all_predicts_np)
        classes = list(map(lambda x: str(x), range(0, self._num_classes)))
        summary = self.confusion_matrix_summary('confusion_matrix', cnf_matrix, classes)
        summary_writer.add_summary(summary, self._global_step)

def MonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=USE_DEFAULT,
                             save_summaries_steps=USE_DEFAULT,
                             save_summaries_secs=USE_DEFAULT,
                             config=None,
                             stop_grace_period_secs=120,
                             log_step_count_steps=100,
                             save_checkpoint_steps=USE_DEFAULT,
                             summary_dir=None,
                             features=None,
                             labels=None):
    if save_summaries_steps == USE_DEFAULT and save_summaries_secs == USE_DEFAULT:
        save_summaries_steps = 100
        save_summaries_secs = None
    elif save_summaries_secs == USE_DEFAULT:
        save_summaries_secs = None
    elif save_summaries_steps == USE_DEFAULT:
        save_summaries_steps = None

    if (save_checkpoint_steps == USE_DEFAULT and
            save_checkpoint_secs == USE_DEFAULT):
        save_checkpoint_steps = None
        save_checkpoint_secs = 600
    elif save_checkpoint_secs == USE_DEFAULT:
        save_checkpoint_secs = None
    elif save_checkpoint_steps == USE_DEFAULT:
        save_checkpoint_steps = None

    scaffold = scaffold or Scaffold()

    all_hooks = []
    if is_chief and chief_only_hooks:
        all_hooks.extend(chief_only_hooks)

    session_creator = ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=checkpoint_dir,
        master=master,
        config=config)

    summary_dir = summary_dir or checkpoint_dir
    if summary_dir:
        if log_step_count_steps and log_step_count_steps > 0:
            all_hooks.append(ImageCounterHook(features, labels, every_n_steps=log_step_count_steps, output_dir=summary_dir))

        if (save_summaries_steps and save_summaries_steps > 0) or (
                save_summaries_secs and save_summaries_secs > 0):
            all_hooks.append(
                basic_session_run_hooks.SummarySaverHook(
                    scaffold=scaffold,
                    save_steps=save_summaries_steps,
                    save_secs=save_summaries_secs,
                    output_dir=summary_dir))

    if checkpoint_dir:
        if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
                save_checkpoint_steps and save_checkpoint_steps > 0):
            all_hooks.append(
                basic_session_run_hooks.CheckpointSaverHook(
                    checkpoint_dir,
                    save_steps=save_checkpoint_steps,
                    save_secs=save_checkpoint_secs,
                    scaffold=scaffold))

    if hooks:
        all_hooks.extend(hooks)

    lp_debug_rank0('all hooks {},\n hooks {},\n chief_only_hooks {},\n checkpoint_dir {}'.format(all_hooks, hooks,
                                                                                           chief_only_hooks,
                                                                                           checkpoint_dir))
    return MonitoredSession(
        session_creator=session_creator,
        hooks=all_hooks,
        stop_grace_period_secs=stop_grace_period_secs)


def _check_listeners_type(saving_listeners):
  """Check listeners type."""
  listeners = list(saving_listeners or [])
  for l in listeners:
    if not isinstance(l, training.CheckpointSaverListener):
      raise TypeError(
          'saving_listeners must be a list of CheckpointSaverListener, '
          'given: {}'.format(l))
  return listeners


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


class HorovodEstimator(estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None):
        super(HorovodEstimator, self).__init__(model_fn=model_fn, model_dir=model_dir, config=config, params=params,
                                               warm_start_from=warm_start_from)

    def train(self,
              input_fn,
              hooks=None,
              steps=None,
              max_steps=None,
              saving_listeners=None):
        if self.config.task_type in (run_config.TaskType.EVALUATOR,
                                     run_config.TaskType.PS):
            raise ValueError(
                'Train has been called wrong configuration. Please use '
                'tf.estimator.train_and_evaluate which calls propper API according '
                'to given configuration. Current configuration: {}.'.format(
                    self.config))

        with context.graph_mode():
            if (steps is not None) and (max_steps is not None):
                raise ValueError('Can not provide both steps and max_steps.')
            if steps is not None and steps <= 0:
                raise ValueError('Must specify steps > 0, given: {}'.format(steps))
            if max_steps is not None and max_steps <= 0:
                raise ValueError(
                    'Must specify max_steps > 0, given: {}'.format(max_steps))

            if max_steps is not None:
                start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
                if max_steps <= start_step:
                    logging.info('Skipping training since max_steps has already saved.')
                    return self

            # lp: avoid reporting type bug
            # hooks = _check_hooks_type(hooks)
            hooks.extend(self._convert_train_steps_to_hooks(steps, max_steps))

            saving_listeners = _check_listeners_type(saving_listeners)
            loss = self._train_model(input_fn, hooks, saving_listeners)
            logging.info('Loss for final step: %s.', loss)
            return self

    def _train_model(self, input_fn, hooks, saving_listeners):
        loss = self._train_model_default(input_fn, hooks, saving_listeners)
        return loss

    def _train_model_default(self, input_fn, hooks, saving_listeners):
        """Initiate training with `input_fn`, without `DistributionStrategies`.

        Args:
          input_fn: A function that provides input data for training as minibatches.
          hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
            callbacks inside the training loop.
          saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
            for callbacks that run immediately before or after checkpoint savings.

        Returns:
          Loss from training
        """

        # lp: add br hook
        worker_hooks = []
        with ops.Graph().as_default() as g, g.device(self._device_fn):
            random_seed.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = self._create_and_assert_global_step(g)

            # Skip creating a read variable if _create_and_assert_global_step
            # returns None (e.g. tf.contrib.estimator.SavedModelEstimator).
            if global_step_tensor is not None:
                training_util._get_or_create_global_step_read(g)  # pylint: disable=protected-access

            features, labels, input_hooks = self._get_features_and_labels_from_input_fn(input_fn,
                                                                                        model_fn_lib.ModeKeys.TRAIN)
            # to-do: add input_hooks here
            worker_hooks.extend(input_hooks)
            estimator_spec = self._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
            global_step_tensor = training_util.get_global_step(g)
            return self._train_with_estimator_spec(estimator_spec, worker_hooks,
                                                   hooks, global_step_tensor,
                                                   saving_listeners, features, labels)

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners, features, labels):
        """Train a model with the given Estimator Spec."""
        if self._warm_start_settings:
            logging.info('Warm-starting with WarmStartSettings: %s' % (self._warm_start_settings,))
            warm_starting_util.warm_start(*self._warm_start_settings)
        # Check if the user created a loss summary, and add one if they didn't.
        # We assume here that the summary is called 'loss'. If it is not, we will
        # make another one with the name 'loss' to ensure it shows up in the right
        # graph in TensorBoard.
        if not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]):
            summary.scalar('loss', estimator_spec.loss)
        ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
        worker_hooks.extend(hooks)
        worker_hooks.extend([
            training.NanTensorHook(estimator_spec.loss)
        ])

        worker_hooks.extend(estimator_spec.training_hooks)

        if not (estimator_spec.scaffold.saver or
                ops.get_collection(ops.GraphKeys.SAVERS)):
            ops.add_to_collection(
                ops.GraphKeys.SAVERS,
                training.Saver(
                    sharded=True,
                    max_to_keep=self._config.keep_checkpoint_max,
                    keep_checkpoint_every_n_hours=(
                        self._config.keep_checkpoint_every_n_hours),
                    defer_build=True,
                    save_relative_paths=True))

        chief_hooks = []
        all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
        saver_hooks = [h for h in all_hooks if isinstance(h, training.CheckpointSaverHook)]
        if (self._config.save_checkpoints_secs or
                self._config.save_checkpoints_steps):
            if not saver_hooks:
                chief_hooks = [
                    training.CheckpointSaverHook(
                        self._model_dir,
                        save_secs=self._config.save_checkpoints_secs,
                        save_steps=self._config.save_checkpoints_steps,
                        scaffold=estimator_spec.scaffold)
                ]
                saver_hooks = [chief_hooks[0]]
        if saving_listeners:
            if not saver_hooks:
                raise ValueError(
                    'There should be a CheckpointSaverHook to use saving_listeners. '
                    'Please set one of the RunConfig.save_checkpoints_steps or '
                    'RunConfig.save_checkpoints_secs.')
            else:
                # It is expected to have one CheckpointSaverHook. If multiple, we pick
                # up the first one to add listener.
                saver_hooks[0]._listeners.extend(saving_listeners)  # pylint: disable=protected-access

        if is_rank0():
            log_step_count_steps = self._config.log_step_count_steps
            checkpoint_dir = self.model_dir
            chief_only_hooks = (tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks))
        else:
            log_step_count_steps = None
            checkpoint_dir = None
            chief_only_hooks = None

        with MonitoredTrainingSession(
                master=self._config.master,
                is_chief=is_rank0(),
                checkpoint_dir=checkpoint_dir,
                scaffold=estimator_spec.scaffold,
                hooks=worker_hooks,
                chief_only_hooks=chief_only_hooks,
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                save_summaries_steps=self._config.save_summary_steps,
                config=self._session_config,
                log_step_count_steps=log_step_count_steps,
                features=features,
                labels=labels) as mon_sess:
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        return loss
