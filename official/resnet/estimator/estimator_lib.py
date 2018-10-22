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
"""Estimator: High level tools for working with models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from official.resnet.estimator.canned.baseline import BaselineClassifier
from official.resnet.estimator.canned.baseline import BaselineRegressor
from official.resnet.estimator.canned.boosted_trees import BoostedTreesClassifier
from official.resnet.estimator.canned.boosted_trees import BoostedTreesRegressor
from official.resnet.estimator.canned.dnn import DNNClassifier
from official.resnet.estimator.canned.dnn import DNNRegressor
from official.resnet.estimator.canned.dnn_linear_combined import DNNLinearCombinedClassifier
from official.resnet.estimator.canned.dnn_linear_combined import DNNLinearCombinedRegressor
from official.resnet.estimator.canned.linear import LinearClassifier
from official.resnet.estimator.canned.linear import LinearRegressor
from official.resnet.estimator.canned.parsing_utils import classifier_parse_example_spec
from official.resnet.estimator.canned.parsing_utils import regressor_parse_example_spec
from official.resnet.estimator.estimator import Estimator
from official.resnet.estimator.estimator import VocabInfo
from official.resnet.estimator.estimator import WarmStartSettings
from official.resnet.estimator.export import export_lib as export
from official.resnet.estimator.exporter import Exporter
from official.resnet.estimator.exporter import FinalExporter
from official.resnet.estimator.exporter import LatestExporter
from official.resnet.estimator.inputs import inputs
from official.resnet.estimator.keras import model_to_estimator
from official.resnet.estimator.model_fn import EstimatorSpec
from official.resnet.estimator.model_fn import ModeKeys
from official.resnet.estimator.run_config import RunConfig
from official.resnet.estimator.training import EvalSpec
from official.resnet.estimator.training import train_and_evaluate
from official.resnet.estimator.training import TrainSpec


# pylint: enable=unused-import,line-too-long,wildcard-import
