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


def is_available():
  """Determine if TensorFlow and the compiler are available."""
  try:
    import tensorflow as tf
  except ModuleNotFoundError:
    logging.warn("Unable to import tensorflow")
    return False
  try:
    import iree.tools.tf.scripts.iree_import_tf.__main__
  except ModuleNotFoundError:
    logging.warning("Unable to find iree-import-tf")
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
  import_type: ImportType = ImportType.OBJECT_GRAPH
  input_type: Union[InputType, str] = InputType.XLA
  saved_model_tags: Set[str] = field(default_factory=set)
  import_extra_args: Sequence[str] = ()
  save_temp_tf_input: Optional[str] = None
  save_temp_mid_level_input: Optional[str] = None
  save_temp_iree_input: Optional[str] = None
  use_tosa: bool = False

  def __post_init__(self):
    self.import_type = ImportType.parse(self.import_type)


def compile_saved_model(saved_model_dir: str, **kwargs):
  """Compiles an on-disk saved model to an IREE binary.

  Args:
    saved_model_dir: Path to directory where the model was saved.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  from iree.tools.tf.scripts.iree_import_tf import __main__
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)

  with tempfile.NamedTemporaryFile(mode="w") as temp_file:
    # Generate MLIR
    __main__.import_saved_model(output_path=temp_file.name,
                                saved_model_dir=saved_model_dir,
                                exported_names=",".join(options.exported_names),
                                import_type=options.import_type.value,
                                tags=",".join(options.saved_model_tags))

    # Full compilation pipeline.
    compile_cl = build_compile_command_line(temp_file.name, tfs, options)
    result = invoke_pipeline([compile_cl])
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
