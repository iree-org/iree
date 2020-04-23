# Lint-as: python3
"""Module init for the python bindings."""

# Copyright 2020 Google LLC
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

# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import

__all__ = [
    # Common
    "Context",
    "Module",
    "CompileOptions",
    "OutputFormat",
    # XLA
    "XLA_IMPORT_PASS_PIPELINE",
    "xla_load_module_proto",
    "xla_compile_module_proto",
]

from typing import Collection, Optional, Sequence

from . import binding

# Native aliases (matches those in the generic compiler).
llvm = binding.llvm
Context = binding.CompilerContext
Module = binding.CompilerModule
CompileOptions = binding.CompileOptions
OutputFormat = binding.OutputFormat

# Pass pipeline that should run to lower a XLA-HLO module to a form suitable
# for input to the IREE compiler.
XLA_IMPORT_PASS_PIPELINE = (
    # Legalize to XLA
    "canonicalize",)


# TODO(suderman): Update PyType to check the xla_computation is an XLA builder.
def xla_load_module_proto(
    xla_computation,
    compiler_context: Optional[Context] = None,
    exported_names: Collection[str] = (),
    pass_pipeline: Sequence[str] = XLA_IMPORT_PASS_PIPELINE) -> Module:
  """Loads a XLA saved model from its persistent representation.

  See also xla_compile_module_proto() for a one-shot API to load and compile.

  Args:
    xla_computation: XLA Computation generate from XLA Python client
    compiler_context: The pyiree.compiler.Context() backing the module.
    exported_names: Optional tuple of strings representing the exported names to
      keep.
    pass_pipeline: Passes to run on the imported module prior to returning.
      Defaults to XLA_IMPORT_PASS_PIPELINE.

  Returns:
    An MLIR Module suitable for compilation by the IREE compiler.
    This can be further compiled to an IREE blob by calling
    .compile_to_sequencer_blob.
  """
  if not compiler_context:
    compiler_context = Context()
  input_module = binding.load_xla_module_proto(
      compiler_context, xla_computation, exported_names=exported_names)
  if pass_pipeline:
    input_module.run_pass_pipeline(pass_pipeline)
  return input_module


def xla_compile_module_proto(
    xla_computation,
    compiler_context: Optional[Context] = None,
    exported_names: Collection[str] = (),
    pass_pipeline: Sequence[str] = XLA_IMPORT_PASS_PIPELINE,
    target_backends: Collection[str] = ()
) -> binding.OpaqueBlob:
  """Loads and compiles a XLA saved model in one shot.

  Args:
    xla_computation: XLA Computation generate from XLA Python client
    compiler_context: The pyiree.compiler.Context() backing the module.
    exported_names: Optional tuple of strings representing the exported names to
      keep.
    pass_pipeline: Passes to run on the imported module prior to returning.
      Defaults to XLA_IMPORT_PASS_PIPELINE.
    target_backends: The specific target backends to compile for (defaults to
      all compiled in targets).

  Returns:
    An OpaqueBlob representing the compiled module.
  """
  input_module = xla_load_module_proto(xla_computation, compiler_context,
                                       exported_names, pass_pipeline)
  return input_module.compile(target_backends=target_backends)
