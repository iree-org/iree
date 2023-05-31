# Lint-as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports TFLite binaries via the `iree-import-tflite` tool."""

from dataclasses import dataclass
from enum import Enum
import logging
import os
import tempfile
from typing import List, Optional, Sequence, Set, Union

from .debugging import TempFileSaver
from .binaries import find_tool, invoke_immediate, invoke_pipeline
from .core import CompilerOptions, DEFAULT_TESTING_BACKENDS, build_compile_command_line

__all__ = [
    "compile_file",
    "compile_str",
    "is_available",
    "DEFAULT_TESTING_BACKENDS",
    "ImportOptions",
]

_IMPORT_TOOL = "iree-import-tflite"


def is_available():
  """Determine if the TFLite frontend is available."""
  try:
    import iree.tools.tflite.scripts.iree_import_tflite.__main__
  except ModuleNotFoundError:
    logging.warning("Unable to find IREE tool iree-import-tflite")
    return False
  return True


@dataclass
class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options.

  Args:
    input_arrays: Sequence of input array node names (if different from
      default).
    output_arrays: Sequence of output array node names (if different from
      default).
    import_only: Only import the module. If True, the result will be textual
      MLIR that can be further fed to the IREE compiler. If False (default),
      the result will be the fully compiled IREE binary. In both cases,
      bytes-like output is returned. Note that if the output_file= is
      specified and import_only=True, then the MLIR form will be written to
      the output file.
    import_extra_args: Extra arguments to pass to the iree-import-tf tool.
    save_temp_tfl_input: Optionally save the IR that results from importing
      the flatbuffer (prior to any further transformations).
    save_temp_iree_input: Optionally save the IR that is the result of the
      import (ready to be passed to IREE).
  """

  input_arrays: Sequence[str] = ()
  output_arrays: Sequence[str] = ()
  import_only: bool = False
  import_extra_args: Sequence[str] = ()
  save_temp_tfl_input: Optional[str] = None
  save_temp_iree_input: Optional[str] = None
  input_type: Optional[str] = "tosa"


def compile_file(fb_path: str, **kwargs):
  """Compiles a TFLite FlatBuffer file to an IREE binary.

  Args:
    fb_path: Path to the FlatBuffer.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  from iree.tools.tflite.scripts.iree_import_tflite import __main__
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)

  with TempFileSaver.implicit() as tfs, tempfile.TemporaryDirectory() as tmpdir:
    if options.import_only and options.output_file:
      # Importing to a file and stopping, write to that file directly.
      tfl_iree_input = options.output_file
    elif options.save_temp_iree_input:
      # Saving the file, use tfs.
      tfl_iree_input = tfs.alloc_optional(
          "tfl-iree-input.mlir", export_as=options.save_temp_iree_input)
    else:
      # Not saving the file, so generate a loose temp file without tfs.
      tfl_iree_input = os.path.join(tmpdir, 'tfl-iree-input.mlir')

    __main__.tflite_to_tosa(flatbuffer=fb_path,
                            bytecode=tfl_iree_input,
                            ordered_input_arrays=options.input_arrays,
                            ordered_output_arrays=options.output_arrays)

    if options.import_only:
      if options.output_file:
        return None
      with open(tfl_iree_input, "r") as f:
        return f.read()

    # Run IREE compilation pipeline
    compile_cl = build_compile_command_line(tfl_iree_input, tfs, options)
    result = invoke_pipeline([compile_cl])
    if options.output_file:
      return None
    return result


def compile_str(input_bytes: bytes, **kwargs):
  """Compiles in-memory TFLite FlatBuffer to an IREE binary.

  Args:
    input_bytes: Flatbuffer content as bytes or IR string.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  input_bytes = input_bytes.encode("utf-8") if isinstance(input_bytes,
                                                          str) else input_bytes
  with tempfile.NamedTemporaryFile(mode="w") as temp_file:
    tempfile.write(input_bytes)
    tempfile.close()
    return compile_file(tempfile.name, **kwargs)
