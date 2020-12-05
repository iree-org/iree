# Lint-as: python3
"""Core compiler interface."""

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

from enum import Enum
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Union

from .tools import *

__all__ = [
    "DEFAULT_TESTING_BACKENDS",
    "compile_file",
    "compile_str",
    "CompilerOptions",
    "OutputFormat",
]

# Default testing backend for invoking the compiler.
DEFAULT_TESTING_BACKENDS = ["vmla"]


class OutputFormat(Enum):
  """The output format of the compiler."""
  FLATBUFFER_BINARY = "flatbuffer-binary"
  FLATBUFFER_TEXT = "flatbuffer-text"
  MLIR_TEXT = "mlir-text"

  @staticmethod
  def parse(spec) -> "OutputFormat":
    if isinstance(spec, OutputFormat):
      return spec
    spec = spec.upper()
    if spec not in OutputFormat.__members__:
      raise ValueError(f"For output_format= argument, expected one of: "
                       f"{', '.join(OutputFormat.__members__.keys())}")
    return OutputFormat[spec]


class CompilerOptions:
  """Options to the compiler backend."""

  def __init__(self,
               *,
               output_file: Optional[str] = None,
               target_backends: Sequence[str] = (),
               output_format: Union[OutputFormat,
                                    str] = OutputFormat.FLATBUFFER_BINARY,
               extra_args: Sequence[str] = (),
               optimize: bool = True,
               strip_debug_ops: bool = False,
               strip_source_map: bool = False,
               strip_symbols: bool = False,
               crash_reproducer_path: Optional[str] = None,
               enable_benchmark: bool = False):
    self.output_file = output_file
    self.target_backends = target_backends
    self.output_format = OutputFormat.parse(output_format)
    self.extra_args = extra_args
    self.optimize = optimize
    self.strip_debug_ops = strip_debug_ops
    self.strip_source_map = strip_source_map
    self.strip_symbols = strip_symbols
    self.crash_reproducer_path = crash_reproducer_path
    self.enable_benchmark = enable_benchmark


def build_compile_command_line(input_file: str,
                               options: CompilerOptions) -> List[str]:
  """Builds a command line for invoking the compiler.

  Args:
    input_file: The input file name.
    options: Compiler options.
  Returns:
    List of strings of command line.
  """
  iree_translate = find_tool("iree-translate")
  if not options.target_backends:
    raise ValueError("Expected a non-empty list for 'target_backends'")

  cl = [
      iree_translate,
      input_file,
      f"--iree-vm-bytecode-module-output-format={options.output_format.value}",
      f"--iree-hal-target-backends={','.join(options.target_backends or [])}",
  ]

  # Output file.
  if options.output_file:
    cl.append(f"-o={options.output_file}")

  # Translation to perform.
  cl.append("--iree-mlir-to-vm-bytecode-module" if not options.enable_benchmark
            else "--iree-mlir-to-executable-benchmark-vm-module")

  # Other options to set if specified.
  if options.strip_debug_ops:
    cl.append("--iree-vm-bytecode-module-strip-debug-ops")
  if options.strip_source_map:
    cl.append("--iree-vm-bytecode-module-strip-source-map")
  if options.strip_symbols:
    cl.append("--iree-vm-bytecode-module-strip-symbols")
  if options.crash_reproducer_path:
    cl.append(
        f"--pass-pipeline-crash-reproducer={options.crash_reproducer_path}")

  cl.extend(options.extra_args)
  return cl


def compile_file(input_file: str, **kwargs):
  """Invokes the IREE compiler on an input file.

  Args:
    input_file: Path to the input file.
    **kwargs: Keyword arguments corresponding to CompilerOptions named tuple.
  Returns:
    Either a byte buffer of the compiled content or None if output_file
    was specified in the options.
  """
  options = CompilerOptions(**kwargs)
  cl = build_compile_command_line(input_file, options)
  result = invoke_immediate(cl)
  if options.output_file:
    return None
  return result


def compile_str(input_str: str, **kwargs):
  """Invokes the IREE compiler with an input string.

  Args:
    input_file: Path to the input file.
    **kwargs: Keyword arguments corresponding to CompilerOptions named tuple.
  Returns:
    Either a byte buffer of the compiled content or None if output_file
    was specified in the options.
  """
  options = CompilerOptions(**kwargs)
  cl = build_compile_command_line("-", options)
  result = invoke_immediate(cl, immediate_input=input_str.encode("utf-8"))
  if options.output_file:
    return None
  return result
