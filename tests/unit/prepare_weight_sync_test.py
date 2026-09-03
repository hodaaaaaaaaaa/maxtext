# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxTextTrainingEngine.prepare_weight_sync single synchronizer logic."""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


import sys
import types as pytypes
import unittest
from unittest import mock

# Ensure tunix C-extension / protobuf initializes before transformers/orbax
try:
  import tunix.experimental.weight_sync.raiden_synchronizer  # pylint: disable=unused-import
except ImportError:
  pass

import jax
import jax.numpy as jnp
from maxtext.training_engine.maxtext_engine import MaxTextTrainingEngine


class PrepareWeightSyncTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Create engine instance without running heavy __init__
    self.engine = MaxTextTrainingEngine.__new__(MaxTextTrainingEngine)
    self.engine._raiden_sync = None
    self.engine._train_step = 0
    self.engine._throttler = mock.MagicMock()
    self.engine._config = pytypes.SimpleNamespace(
        scan_layers=False,
        num_decoder_layers=2,
        param_scan_axis=1,
        weight_sync_debug=False,
    )
    self.engine._use_weight_converter = True
    self.engine._weight_converter = mock.MagicMock()
    self.engine._rollout_backend = "maxtext"
    self.engine._get_trainable_params_state = mock.MagicMock(return_value={"layer": jnp.zeros((4, 4))})

  def _make_dummy_metadata(self, num_vars=2):
    meta = mock.MagicMock()
    meta.variables = [f"var_{i}" for i in range(num_vars)]
    meta.mesh_axes = (1, 1)
    return meta

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_single_synchronizer_creation_and_binding(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.active = True
    mock_sync.work_unit_metadata.return_value = self._make_dummy_metadata(num_vars=10)
    mock_sync.checksums.return_value = {}
    mock_sync_cls.return_value = mock_sync

    converted = {"converted_layer": jnp.zeros((4, 4))}
    self.engine._weight_converter.convert.return_value = converted

    metadata = self.engine.prepare_weight_sync()

    self.assertEqual(len(metadata), 1)
    self.assertIs(self.engine._raiden_sync, mock_sync)
    mock_sync_cls.assert_called_once_with(
        job_name="trainer",
        worker_index=jax.process_index(),
        auto_h2d=False,
        parallelism=4,
    )

    self.engine._weight_converter.convert.assert_called_once()
    mock_sync.bind.assert_called_once_with(converted)
    mock_sync.d2h.assert_called_once()
    mock_sync.work_unit_metadata.assert_called_once()

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_rebind_reuses_single_sync_instance(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.active = True
    mock_sync.work_unit_metadata.return_value = self._make_dummy_metadata(num_vars=10)
    mock_sync_cls.return_value = mock_sync

    # Round 1
    self.engine._weight_converter.convert.return_value = {"p": 0}
    self.engine.prepare_weight_sync()
    self.assertEqual(mock_sync_cls.call_count, 1)

    # Round 2
    self.engine._train_step = 1
    self.engine._weight_converter.convert.return_value = {"p": 1}
    self.engine.prepare_weight_sync()

    # Still only 1 synchronizer instance created
    self.assertEqual(mock_sync_cls.call_count, 1)
    self.assertEqual(mock_sync.bind.call_count, 2)


if __name__ == "__main__":
  unittest.main()
