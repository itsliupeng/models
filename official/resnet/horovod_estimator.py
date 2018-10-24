from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.training.monitored_session import USE_DEFAULT, Scaffold, MonitoredSession, ChiefSessionCreator
import tensorflow as tf

import horovod.tensorflow as hvd

estimator.Estimator._assert_members_are_not_overridden = lambda _: None


def is_rank0():
    return hvd.rank() == 0


def lp_debug(msg):
    tf.logging.info('lp-debug: ' + msg)


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, device=''):
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

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = hvd.broadcast_global_variables(self.root_rank)

    def after_create_session(self, session, coord):
        lp_debug('br begin')
        session.run(self.bcast_op)
        lp_debug('br end')


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
                             summary_dir=None):
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

    # lp-to-do: restore from checkpoint, should br rank 0 variables
    session_creator = ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=checkpoint_dir,
        master=master,
        config=config)

    summary_dir = summary_dir or checkpoint_dir
    if summary_dir:
        if log_step_count_steps and log_step_count_steps > 0:
            all_hooks.append(
                basic_session_run_hooks.StepCounterHook(
                    output_dir=summary_dir, every_n_steps=log_step_count_steps))

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

    tf.logging.info('lp: rank {}, all hooks {}, hooks {}, chief_only_hooks {} checkpoint_dir {}'.format(hvd.rank(), all_hooks, hooks, chief_only_hooks, checkpoint_dir))
    return MonitoredSession(
        session_creator=session_creator,
        hooks=all_hooks,
        stop_grace_period_secs=stop_grace_period_secs)


class HorovodEstimator(estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None):
        super(HorovodEstimator, self).__init__(model_fn=model_fn, model_dir=model_dir, config=config, params=params,
                                               warm_start_from=warm_start_from)

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
        worker_hooks = [BroadcastGlobalVariablesHook(0)]
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
                                                   saving_listeners)

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners):
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
        worker_hooks.append(
            training.NanTensorHook(estimator_spec.loss)
        )
        if self._config.log_step_count_steps is not None:
            if is_rank0():
                worker_hooks.append(
                    training.LoggingTensorHook(
                        {
                            'loss': estimator_spec.loss,
                            'step': global_step_tensor
                        },
                        every_n_iter=self._config.log_step_count_steps)
                )
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
        saver_hooks = [
            h for h in all_hooks if isinstance(h, training.CheckpointSaverHook)]
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
                log_step_count_steps=log_step_count_steps) as mon_sess:
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        return loss
