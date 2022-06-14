# Lint-as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports XLA artifacts via the `iree-import-xla` tool."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
import tempfile
from typing import List, Optional, Sequence, Set, Union

from .core import CompilerOptions, DEFAULT_TESTING_BACKENDS, build_compile_command_line
from .debugging import TempFileSaver
from .binaries import find_tool, invoke_immediate, invoke_pipeline

__all__ = [
    "compile_file",
    "compile_str",
    "is_available",
    "DEFAULT_TESTING_BACKENDS",
    "ImportOptions",
    "ImportFormat",
]

_IMPORT_TOOL = "iree-import-xla"


def is_available():
  """Determine if the XLA frontend is available."""
  try:
    find_tool(_IMPORT_TOOL)
  except ValueError:
    logging.warning("Unable to find IREE tool %s", _IMPORT_TOOL)
    return False
  return True


class ImportFormat(Enum):
  """Import type of the model."""
  BINARY_PROTO = "binary_proto"
  TEXT_PROTO = "text_proto"
  HLO_TEXT = "hlo_text"
  MLIR_TEXT = "mlir_text"

  @staticmethod
  def parse(spec: Union[str, ImportFormat]) -> ImportFormat:
    """Parses or returns an ImportFormat.

    Args:
      spec: An ImportFormat instance or the case-insensitive name of one of
        the enum values.
    Returns:
      An ImportFormat instance.
    """
    if isinstance(spec, ImportFormat):
      return spec
    spec = spec.upper()
    if spec not in ImportFormat.__members__:
      raise ValueError(f"For import_format= argument, expected one of: "
                       f"{', '.join(ImportFormat.__members__.keys())}")
    return ImportFormat[spec]


@dataclass
class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options.

  Args:
    import_format: Format of the proto (text or binary).
    save_temp_iree_input: Optionally save the IR that is the result of the
      import (ready to be passed to IREE).
  """

  import_only: bool = False
  import_format: Union[ImportFormat, str] = ImportFormat.BINARY_PROTO
  import_extra_args: Sequence[str] = ()
  save_temp_mhlo_input: Optional[str] = None
  save_temp_iree_input: Optional[str] = None

  def __post_init__(self):
    self.import_format = ImportFormat.parse(self.import_format)


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
      f"--xla-format={options.import_format.value}",
  ]

  if options.import_only and options.output_file:
    # Import stage directly outputs.
    output_file = tfs.alloc_optional("xla-output.mlir",
                                     export_as=options.output_file)
    cl.append(f"-o={output_file}")

  # MLIR flags.
  if options.output_mlir_debuginfo:
    cl.append("--mlir-print-debuginfo")
  if options.output_generic_mlir:
    cl.append("--mlir-print-op-generic")

  # Save temps flags.
  save_mhlo_input = tfs.alloc_optional("tf-mhlo.mlir",
                                       export_as=options.save_temp_mhlo_input)
  if save_mhlo_input:
    cl.append(f"--save-temp-mhlo-input={save_mhlo_input}")
  iree_input = tfs.alloc_optional("xla-iree-input.mlir",
                                  export_as=options.save_temp_iree_input)
  if iree_input:
    cl.append(f"--save-temp-iree-input={iree_input}")

  # Crash reproducer (locally qualified).
  requested_crash_reproducer_path = options.crash_reproducer_path
  if requested_crash_reproducer_path:
    requested_crash_reproducer_path = (requested_crash_reproducer_path +
                                       ".import-xla")
  crash_reproducer_path = tfs.alloc_optional(
      "xla-reproducer.mlir", export_as=requested_crash_reproducer_path)
  if crash_reproducer_path:
    cl.append(f"--mlir-pass-pipeline-crash-reproducer={crash_reproducer_path}")

  # Extra args.
  cl.extend(options.import_extra_args)
  return cl


def compile_file(xla_file_path: str, **kwargs):
  """Compiles an on-disk XLA protocol buffer to an IREE binary.

  Args:
    xla_file_path: Path to the XLA proto file.
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)
    import_cl = build_import_command_line(xla_file_path, tfs, options)
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


def compile_str(xla_content: Union[bytes, str], **kwargs):
  """Compiles in-memory XLA content to an IREE binary.

  Args:
    xla_content: Either bytes or str content (str is only valid for text
      formats).
    **kwargs: Keyword args corresponding to ImportOptions or CompilerOptions.
  Returns:
    A bytes-like object with the compiled output or None if output_file=
    was specified.
  """
  with TempFileSaver.implicit() as tfs:
    options = ImportOptions(**kwargs)
    if isinstance(xla_content, str):
      if options.import_format not in [
          ImportFormat.TEXT_PROTO, ImportFormat.HLO_TEXT, ImportFormat.MLIR_TEXT
      ]:
        raise ValueError("If passing a string, ImportFormat must be TEXT_PROTO")
      xla_content = xla_content.encode("utf-8")

    import_cl = build_import_command_line("-", tfs, options)
    if options.import_only:
      # One stage tool pipeline.
      result = invoke_immediate(import_cl, immediate_input=xla_content)
      if options.output_file:
        return None
      return result

    # Full compilation pipeline.
    compile_cl = build_compile_command_line("-", tfs, options)
    result = invoke_pipeline([import_cl, compile_cl],
                             immediate_input=xla_content)
    if options.output_file:
      return None
    return result
