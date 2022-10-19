# Lint-as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports TensorFlow artifacts via the `iree-import-tf tool."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
import tempfile
from typing import List, Optional, Sequence, Set, Union

from .core import (
    CompilerOptions,
    DEFAULT_TESTING_BACKENDS,
    InputType,
    build_compile_command_line,
)
from .debugging import TempFileSaver
from .binaries import find_tool, invoke_immediate, invoke_pipeline

__all__ = [
    "compile_saved_model",
    "compile_module",
    "is_available",
    "DEFAULT_TESTING_BACKENDS",
    "ImportOptions",
    "ImportType",
]

_TF_IMPORT_TOOL = "iree-import-tf"


def is_available():
  """Determine if TensorFlow and the compiler are available."""
  try:
    import tensorflow as tf
  except ModuleNotFoundError:
    logging.warn("Unable to import tensorflow")
    return False
  try:
    find_tool(_TF_IMPORT_TOOL)
  except ValueError:
    logging.warning("Unable to find IREE tool %s", _TF_IMPORT_TOOL)
    return False
  return True


class ImportType(Enum):
  """Import type of the model."""
  OBJECT_GRAPH = "savedmodel_v2"
  V2 = "savedmodel_v2"
  SIGNATURE_DEF = "savedmodel_v1"
  V1 = "savedmodel_v1"

  @staticmethod
  def parse(spec: Union[str, ImportType]) -> ImportType:
    """Parses or returns an ImportType.

    Args:
      spec: An ImportType instance or the case-insensitive name of one of
        the enum values.
    Returns:
      An ImportType instance.
    """
    if isinstance(spec, ImportType):
      return spec
    spec = spec.upper()
    if spec not in ImportType.__members__:
      raise ValueError(f"For import_type= argument, expected one of: "
                       f"{', '.join(ImportType.__members__.keys())}")
    return ImportType[spec]


@dataclass
class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options.

  Args:
    exported_names: Optional sequence representing the exported names to
      keep (object graph/v2 models only).
    import_only: Only import the module. If True, the result will be textual
      MLIR that can be further fed to the IREE compiler. If False (default),
      the result will be the fully compiled IREE binary. In both cases,
      bytes-like output is returned. Note that if the output_file= is
      specified and import_only=True, then the MLIR form will be written to
      the output file.
    import_type: Type of import to perform. See ImportType enum.
    saved_model_tags: Set of tags to export (signature def/v1 saved models
      only).
    import_extra_args: Extra arguments to pass to the iree-import-tf tool.
    save_temp_tf_input: Optionally save the IR that is input to the
      TensorFlow pipeline.
    save_temp_mid_level_input: Optionally save the IR that is input to the
      mid level IR.
    save_temp_iree_input: Optionally save the IR that is the result of the
      import (ready to be passed to IREE).
  """

  exported_names: Sequence[str] = ()
  import_only: bool = False
  import_type: Union[ImportType, str] = ImportType.OBJECT_GRAPH
  input_type: Union[InputType, str] = InputType.XLA
  saved_model_tags: Set[str] = field(default_factory=set)
  import_extra_args: Sequence[str] = ()
  save_temp_tf_input: Optional[str] = None
  save_temp_mid_level_input: Optional[str] = None
  save_temp_iree_input: Optional[str] = None
  use_tosa: bool = False

  def __post_init__(self):
    self.import_type = ImportType.parse(self.import_type)


def build_import_command_line(input_path: str, tfs: TempFileSaver,
                              options: ImportOptions) -> List[str]:
  """Builds a command line for invoking the import stage.

  Args:
    input_path: The input path.
    tfs: TempFileSaver.
    options: Import options.
  Returns:
    List of strings of command line.
  """
  tf_import = find_tool(_TF_IMPORT_TOOL)
  cl = [
      tf_import,
      input_path,
      f"--tf-import-type={options.import_type.value}",
      f"--tf-savedmodel-exported-names={','.join(options.exported_names)}",
      f"--tf-savedmodel-tags={','.join(options.saved_model_tags)}",
  ]

  if options.import_only and options.output_file:
    # Import stage directly outputs.
    output_file = tfs.alloc_optional("tf-output.mlir",
                                     export_as=options.output_file)
    cl.append(f"-o={output_file}")
    cl.append(f"--output-format=mlir-ir")

  # MLIR flags.
  if options.output_mlir_debuginfo:
    cl.append("--mlir-print-debuginfo")
  if options.output_generic_mlir:
    cl.append("--mlir-print-op-generic")

  # Save temps flags.
  save_tf_input = tfs.alloc_optional("tf-input.mlir",
                                     export_as=options.save_temp_tf_input)
  if save_tf_input:
    cl.append(f"--save-temp-tf-input={save_tf_input}")
  save_mid_level_input = tfs.alloc_optional(
      "tf-mid-level-input.mlir", export_as=options.save_temp_mid_level_input)
  if save_mid_level_input:
    cl.append(f"--save-temp-mid-level-input={save_mid_level_input}")
  save_iree_input = tfs.alloc_optional("tf-iree-input.mlir",
                                       export_as=options.save_temp_iree_input)
  if save_iree_input:
    cl.append(f"--save-temp-iree-input={save_iree_input}")

  if options.use_tosa:
    cl.append(f"--use-tosa")

  # Crash reproducer (locally qualified).
  requested_crash_reproducer_path = options.crash_reproducer_path
  if requested_crash_reproducer_path:
    requested_crash_reproducer_path = (requested_crash_reproducer_path +
                                       ".import-tf")
  crash_reproducer_path = tfs.alloc_optional(
      "tf-reproducer.mlir", export_as=requested_crash_reproducer_path)
  if crash_reproducer_path:
    cl.append(f"--mlir-pass-pipeline-crash-reproducer={crash_reproducer_path}")

  # Extra args.
  cl.extend(options.import_extra_args)
  return cl


def compile_saved_model(saved_model_dir: str, **kwargs):
  """Compiles an on-disk saved model to an IREE binary.

  Args:
    saved_model_dir: Path to directory where the model was saved.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)
    import_cl = build_import_command_line(saved_model_dir, tfs, options)
    if options.import_only:
      # One stage tool pipeline.
      result = invoke_immediate(import_cl)
      if options.output_file:
        return None
      return result

    # Full compilation pipeline.
    compile_cl = build_compile_command_line("-", tfs, options)
    result = invoke_pipeline([import_cl, compile_cl])
    if options.output_file:
      return None
    return result


def compile_module(module, saved_model_dir: Optional[str] = None, **kwargs):
  """Compiles a tf.Module to an IREE binary (by saving to disk).

  Args:
    module: The tf.Module instance to convert to MLIR
    saved_model_dir: Optional path to save the tf.Module to. The module will not
      be persisted on disk outside of this call if this is not provided.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    Same as compile_saved_model().
  """
  with TempFileSaver.implicit() as tfs:

    def do_it(saved_model_dir):
      import tensorflow as tf
      options = tf.saved_model.SaveOptions(save_debug_info=True)
      tf.saved_model.save(module, saved_model_dir, options=options)
      return compile_saved_model(saved_model_dir, **kwargs)

    if saved_model_dir:
      return do_it(saved_model_dir)
    else:
      with tempfile.TemporaryDirectory(suffix=".sm") as td:
        return do_it(td)
