# Lint-as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports TFLite binaries via the `iree-import-tflite` tool."""
# TODO(#4131) python>=3.7: Use postponed type annotations.

from enum import Enum
import logging
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
  """Determine if the XLA frontend is available."""
  try:
    find_tool(_IMPORT_TOOL)
  except ValueError:
    logging.warning("Unable to find IREE tool %s", _IMPORT_TOOL)
    return False
  return True


# TODO(#4131) python>=3.7: Consider using a dataclass.
class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options."""

  def __init__(self,
               input_arrays: Sequence[str] = (),
               output_arrays: Sequence[str] = (),
               import_only: bool = False,
               import_extra_args: Sequence[str] = (),
               save_temp_tfl_input: Optional[str] = None,
               save_temp_iree_input: Optional[str] = None,
               input_type: Optional[str] = "tosa",
               **kwargs):
    """Initialize options from keywords.

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
    super().__init__(input_type=input_type, **kwargs)
    self.input_arrays = input_arrays
    self.output_arrays = output_arrays
    self.import_only = import_only
    self.import_extra_args = import_extra_args
    self.save_temp_tfl_input = save_temp_tfl_input
    self.save_temp_iree_input = save_temp_iree_input


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
  import_tool = find_tool(_IMPORT_TOOL)
  cl = [
      import_tool,
      input_path,
  ]

  if options.import_only and options.output_file:
    # Import stage directly outputs.
    output_file = tfs.alloc_optional("tflite-output.mlir",
                                     export_as=options.output_file)
    cl.append(f"-o={options.output_file}")

  # Input arrays.
  if options.input_arrays:
    for input_array in options.input_arrays:
      cl.append(f"--input-array={input_array}")
    for output_array in options.output_arrays:
      cl.append(f"--output-array={output_array}")

  # MLIR flags.
  if options.output_mlir_debuginfo:
    cl.append("--mlir-print-debuginfo")
  if options.output_generic_mlir:
    cl.append("--mlir-print-op-generic")

  # Save temps flags.
  tfl_input = tfs.alloc_optional("tflite-input.mlir",
                                 export_as=options.save_temp_tfl_input)
  if tfl_input:
    cl.append(f"--save-temp-tfl-input={tfl_input}")
  iree_input = tfs.alloc_optional("tflite-iree-input.mlir",
                                  export_as=options.save_temp_iree_input)
  if iree_input:
    cl.append(f"--save-temp-iree-input={iree_input}")

  # Crash reproducer (locally qualified).
  requested_crash_reproducer_path = options.crash_reproducer_path
  if requested_crash_reproducer_path:
    requested_crash_reproducer_path = (requested_crash_reproducer_path +
                                       ".import-tflite")
  crash_reproducer_path = tfs.alloc_optional(
      "tflite-reproducer.mlir", export_as=requested_crash_reproducer_path)
  if crash_reproducer_path:
    cl.append(f"--mlir-pass-pipeline-crash-reproducer={crash_reproducer_path}")

  # Extra args.
  cl.extend(options.import_extra_args)
  return cl


def compile_file(fb_path: str, **kwargs):
  """Compiles a TFLite flatbuffer file to an IREE binary.

  Args:
    fb_path: Path to the flatbuffer.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)
    import_cl = build_import_command_line(fb_path, tfs, options)
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


def compile_str(fb_content: bytes, **kwargs):
  """Compiles in-memory TFLite flatbuffer to an IREE binary.

  Args:
    fb_content: Flatbuffer content as bytes.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)
    import_cl = build_import_command_line("-", tfs, options)
    if options.import_only:
      # One stage tool pipeline.
      result = invoke_immediate(import_cl, immediate_input=fb_content)
      if options.output_file:
        return None
      return result

    # Full compilation pipeline.
    compile_cl = build_compile_command_line("-", tfs, options)
    result = invoke_pipeline([import_cl, compile_cl],
                             immediate_input=fb_content)
    if options.output_file:
      return None
    return result
