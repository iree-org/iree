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
"""High level compiler API.

This imports parts of the native bindings as appropriate.
"""

__all__ = [
    "TF_IMPORT_PASS_PIPELINE",
    "tf_load_saved_model",
    "tf_compile_saved_model",
]

from typing import Collection, Optional, Sequence

from . import binding as _binding
from .binding import CompilerContext as Context
from .binding import CompilerModule as Module

# Pass pipeline that should run to lower a TF saved_model to a form suitable
# for input to the IREE compiler.
TF_IMPORT_PASS_PIPELINE = (
    # Clean up tf_executor and extraneous unused functions.
    "tf-saved-model-delete-unused-funcs",
    "tf-executor-graph-pruning",
    "tf-standard-pipeline",
    "canonicalize",

    # Clean up control flow
    "tf-functional-control-flow-to-cfg",
    "inline",
    "tf-saved-model-delete-unused-funcs",
    "canonicalize",

    # Legalize to XLA
    "xla-legalize-tf{allow-partial-conversion=true}",
    "canonicalize",

    # Now that the IR is starting to look nice, optimize global tensors.
    "tf-saved-model-optimize-global-tensors",

    # Adopt saved_model exports into IREE.
    "iree-tf-saved-model-adopt-exports",
)


def tf_load_saved_model(
    saved_model_dir: str,
    compiler_context: Optional[Context] = None,
    exported_names: Collection[str] = (),
    pass_pipeline: Sequence[str] = TF_IMPORT_PASS_PIPELINE) -> Module:
  """Loads a TensorFlow saved model from its persistent representation.

  See also tf_compile_saved_model() for a one-shot API to load and compile.

  Args:
    saved_model_dir: Directory of the saved model.
    compiler_context: The pyiree.compiler.Context() backing the module.
    exported_names: Optional tuple of strings representing the exported names to
      keep.
    pass_pipeline: Passes to run on the imported module prior to returning.
      Defaults to TF_IMPORT_PASS_PIPELINE.

  Returns:
    An MLIR Module suitable for compilation by the IREE compiler.
    This can be further compiled to an IREE blob by calling
    .compile_to_sequencer_blob.
  """
  if not compiler_context:
    compiler_context = Context()
  input_module = _binding.tf_interop.load_saved_model(
      compiler_context, saved_model_dir, exported_names=exported_names)
  if pass_pipeline:
    input_module.run_pass_pipeline(pass_pipeline)
  return input_module


def tf_compile_saved_model(
    saved_model_dir: str,
    compiler_context: Optional[Context] = None,
    exported_names: Collection[str] = (),
    pass_pipeline: Sequence[str] = TF_IMPORT_PASS_PIPELINE,
    target_backends: Collection[str] = ()
) -> _binding.OpaqueBlob:
  """Loads and compiles a TensorFlow saved model in one shot.

  Args:
    saved_model_dir: Directory of the saved model.
    compiler_context: The pyiree.compiler.Context() backing the module.
    exported_names: Optional tuple of strings representing the exported names to
      keep.
    pass_pipeline: Passes to run on the imported module prior to returning.
      Defaults to TF_IMPORT_PASS_PIPELINE.
    target_backends: The specific target backends to compile for (defaults to
      all compiled in targets).

  Returns:
    An OpaqueBlob representing the compiled module.
  """
  input_module = tf_load_saved_model(saved_model_dir, compiler_context,
                                     exported_names, pass_pipeline)
  return input_module.compile(target_backends=target_backends)
