# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utilities interop with TensorFlow."""

# pylint: disable=not-callable
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import os
import random
import re
import tempfile

from absl import flags
from absl import logging
import numpy as np
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


def set_random_seed(seed=0):
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def compile_tf_module(tf_module,
                      exported_names=(),
                      target_backends=(),
                      artifacts_dir=None,
                      keep_saved_model=False):
  """Compiles a TensorFlow tf.Module and optionally saves compilation artifacts.

  Args:
    tf_module: A tf.Module.
    exported_names: Iterable of dotted function names to consider for
      compilation.
    target_backends: Iterable of string backend names to compile for.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved.
    keep_saved_model: An optional boolean controlling whether or not to keep the
      saved model used for translating the tf_module to a compiler_module in the
      specified artifacts_dir.

  Returns:
    An _IreeCompiledModule.
  """

  def _compile_from_path(sm_path):
    """Helper function for compile_tf_module."""
    # We break up the compilation here so we can save intermediary artifacts.
    compiler_context = compiler.Context()

    if artifacts_dir is not None:
      normalized_backends = []
      for backend in target_backends:
        # Remove unusual characters and ensure names don't end or start in "_".
        backend = re.sub("[^0-9a-zA-Z_]+", "_", backend)
        normalized_backends.append(backend.strip("_"))
      backends_string = "__".join(normalized_backends)

    # Convert the tf_module into raw TF input MLIR.
    compiler_module = compiler.tf_load_saved_model(
        sm_path,
        exported_names=exported_names,
        compiler_context=compiler_context,
        pass_pipeline=())

    if artifacts_dir is not None:
      tf_mlir_path = os.path.join(artifacts_dir,
                                  f"tf_input__{backends_string}.mlir")
      logging.info("Saving raw TF input MLIR to: %s", tf_mlir_path)
      with open(tf_mlir_path, "w") as f:
        f.write(compiler_module.to_asm())

    # Now run the passes manually that tf_load_saved_model would usually do.
    compiler_module.run_pass_pipeline(compiler.TF_IMPORT_PASS_PIPELINE)

    if artifacts_dir is not None:
      iree_mlir_path = os.path.join(artifacts_dir,
                                    f"iree_input__{backends_string}.mlir")
      logging.info("Saving IREE input MLIR to: %s", iree_mlir_path)
      with open(iree_mlir_path, "w") as f:
        f.write(compiler_module.to_asm())

    compiled_module = compiler_module.compile(target_backends=target_backends)
    if artifacts_dir is not None:
      compiled_path = os.path.join(artifacts_dir,
                                   f"compiled__{backends_string}.vmfb")
      logging.info("Saving compiled IREE module to: %s", compiled_path)
      with open(compiled_path, "wb") as f:
        f.write(compiled_module)

    return compiled_module

  options = tf.saved_model.SaveOptions(save_debug_info=True)
  if not keep_saved_model or artifacts_dir is None:
    # Round-trip through a temporary directory.
    with tempfile.TemporaryDirectory() as sm_path:
      tf.saved_model.save(tf_module, sm_path, options=options)
      return _compile_from_path(sm_path)
  else:
    # Use the supplied directory.
    sm_path = os.path.join(artifacts_dir, "saved_model")
    tf.saved_model.save(tf_module, sm_path, options=options)
    return _compile_from_path(sm_path)
