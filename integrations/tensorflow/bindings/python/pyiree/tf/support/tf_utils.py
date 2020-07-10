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
"""Utilities interop with TensorFlow."""

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
                      target_backends=(),
                      exported_names=(),
                      artifacts_dir=None):
  """Compiles a TensorFlow tf.Module and optionally saves compilation artifacts.

  If artifacts_dir is provided then the following artifacts will be saved:
    saved_model:
      A TF SavedModel directory containing the files used translate the
      tf.Module into an IREE module.
    tf_input__backends.mlir:
      MLIR for the module in TF's input dialect.
    iree_input__backends.mlir:
      The MLIR above translated to IREE via compiler.TF_IMPORT_PASS_PIPELINE.
    compiled__backends.vmfb:
      A VM FlatBuffer compiled to the target backends from the IREE MLIR above.
  Here 'backends' is a '__' delimited list of iree backends (e.g. vmla__llvm_ir)

  Args:
    tf_module: A tf.Module.
    target_backends: Iterable of string backend names to compile for.
    exported_names: Iterable of dotted function names to consider for
      compilation.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved.

  Returns:
    A compiled IREE module blob.
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
  if artifacts_dir is not None:
    # Save the saved model alongside the other compilation artifacts.
    sm_path = os.path.join(artifacts_dir, "saved_model")
    tf.saved_model.save(tf_module, sm_path, options=options)
    return _compile_from_path(sm_path)
  else:
    # Round-trip the saved model through a temporary directory.
    with tempfile.TemporaryDirectory() as sm_path:
      tf.saved_model.save(tf_module, sm_path, options=options)
      return _compile_from_path(sm_path)
