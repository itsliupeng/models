from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import tempfile

import numpy as np
import six


from tensorflow import estimator
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.estimator.export import export as export_helpers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.summary import summary
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import device_setter
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.util import compat
from tensorflow.python.util import compat_internal
from tensorflow.python.util import function_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import estimator_export


class HorovodEstimator(estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None):
        super(HorovodEstimator, self).__init__(model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

    def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks,
                                   global_step_tensor, saving_listeners):
        """Train a model with the given Estimator Spec."""
        if self._warm_start_settings:
            logging.info('Warm-starting with WarmStartSettings: %s' %
                         (self._warm_start_settings,))
            warm_starting_util.warm_start(*self._warm_start_settings)
        # Check if the user created a loss summary, and add one if they didn't.
        # We assume here that the summary is called 'loss'. If it is not, we will
        # make another one with the name 'loss' to ensure it shows up in the right
        # graph in TensorBoard.
        if not any([x.op.name == 'loss'
                    for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]):
            summary.scalar('loss', estimator_spec.loss)
        ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
        worker_hooks.extend(hooks)
        worker_hooks.append(
            training.NanTensorHook(estimator_spec.loss)
        )
        if self._config.log_step_count_steps is not None:
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
        with training.MonitoredTrainingSession(
                master=self._config.master,
                is_chief=self._config.is_chief,
                checkpoint_dir=self._model_dir,
                scaffold=estimator_spec.scaffold,
                hooks=worker_hooks,
                chief_only_hooks=(
                        tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks)),
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                save_summaries_steps=self._config.save_summary_steps,
                config=self._session_config,
                log_step_count_steps=self._config.log_step_count_steps) as mon_sess:
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        return loss


