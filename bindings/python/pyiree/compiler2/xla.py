# Lint-as: python3
"""XLA compiler interface."""

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

# TODO(#4131) python>=3.7: Use postponed type annotations.

from enum import Enum
import logging
import tempfile
from typing import List, Optional, Sequence, Set, Union

from .tools import find_tool, invoke_immediate, invoke_pipeline
from .core import CompilerOptions, DEFAULT_TESTING_BACKENDS, build_compile_command_line

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

  @staticmethod
  def parse(spec: Union[str, "ImportFormat"]) -> "ImportFormat":
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


# TODO(#4131) python>=3.7: Consider using a dataclass.
class ImportOptions(CompilerOptions):
  """Import options layer on top of the backend compiler options."""

  def __init__(self,
               import_only: bool = False,
               import_format: Union[ImportFormat,
                                    str] = ImportFormat.BINARY_PROTO,
               import_extra_args: Sequence[str] = (),
               save_temp_iree_input: Optional[str] = None,
               **kwargs):
    """Initialize options from keywords.

    Args:
      import_format: Format of the proto (text or binary).
      save_temp_iree_input: Optionally save the IR that is the result of the
        import (ready to be passed to IREE).
    """
    super().__init__(**kwargs)
    self.import_only = import_only
    self.import_format = ImportFormat.parse(import_format)
    self.import_extra_args = import_extra_args
    self.save_temp_iree_input = save_temp_iree_input


def build_import_command_line(input_path: str,
                              options: ImportOptions) -> List[str]:
  """Builds a command line for invoking the import stage.

  Args:
    input_path: The input path.
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
    if options.output_file:
      cl.append(f"-o={options.output_file}")
  # Save temps flags.
  if options.save_temp_iree_input:
    cl.append(f"--save-temp-iree-input={options.save_temp_iree_input}")
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
  options = ImportOptions(**kwargs)
  import_cl = build_import_command_line(xla_file_path, options)
  if options.import_only:
    # One stage tool pipeline.
    result = invoke_immediate(import_cl)
    if options.output_file:
      return None
    return result

  # Full compilation pipeline.
  compile_cl = build_compile_command_line("-", options)
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
  options = ImportOptions(**kwargs)
  if isinstance(xla_content, str):
    if options.import_format not in [
        ImportFormat.TEXT_PROTO, ImportFormat.HLO_TEXT
    ]:
      raise ValueError("If passing a string, ImportFormat must be TEXT_PROTO")
    xla_content = xla_content.encode("utf-8")

  import_cl = build_import_command_line("-", options)
  if options.import_only:
    # One stage tool pipeline.
    result = invoke_immediate(import_cl, immediate_input=xla_content)
    if options.output_file:
      return None
    return result

  # Full compilation pipeline.
  compile_cl = build_compile_command_line("-", options)
  result = invoke_pipeline([import_cl, compile_cl], immediate_input=xla_content)
  if options.output_file:
    return None
  return result
