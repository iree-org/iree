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

# TODO(#4131) python>=3.7: Use postponed type annotations.

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
# TODO: Remove these. In the absence of default profiles, though, it is better
# to centralize.
DEFAULT_TESTING_BACKENDS = ["dylib-llvm-aot"]
DEFAULT_TESTING_DRIVER = "dylib"


class OutputFormat(Enum):
  """The output format of the compiler."""
  FLATBUFFER_BINARY = "flatbuffer-binary"
  FLATBUFFER_TEXT = "flatbuffer-text"
  MLIR_TEXT = "mlir-text"

  @staticmethod
  def parse(spec: Union[str, "OutputFormat"]) -> "OutputFormat":
    """Parses or returns an OutputFormat.

    Args:
      spec: An OutputFormat instance or the case-insensitive name of one of
        the enum values.
    Returns:
      An OutputFormat instance.
    """
    if isinstance(spec, OutputFormat):
      return spec
    spec = spec.upper().replace("-", "_")
    if spec not in OutputFormat.__members__:
      raise ValueError(f"For output_format= argument, expected one of: "
                       f"{', '.join(OutputFormat.__members__.keys())}")
    return OutputFormat[spec]


# TODO(#4131) python>=3.7: Consider using a dataclass.
class CompilerOptions:
  """Options to the compiler backend.

  Keyword options:
    output_file: Optionally save the compiled binary to a file instead of
      returning it.
    target_backends: List of str names of target backends to compile into
      the binary. The resulting binary will run on targets that match one
      or more of the compiled backends.
    output_format: Override the output format. See the OutputFormat enum.
      Values can either be an enum value or a case-insensitive name of
      the option. Typically used for debugging
    extra_args: Optional list of additional arguments to pass to the compiler.
      Example: ["--print-ir-after-all"]
    optimize: Whether to apply some default high level optimizations (default
      True).
    output_mlir_debuginfo: Include debuginfo (including paths) in any saved or
      returned MLIR.
    output_generic_mlir: Use the generic (and more portable) MLIR formatting for
      any saved or returned MLIR instead of the per-dialect custom assembly.
    extended_diagnostics: Outputs extended information on diagnostics,
      potentially outputting very verbosely (defaults to False).
    strip_debug_ops: Whether to strip high level operations used to aid
      debugging.
    strip_source_map: Whether to strip source map information (used to generate
      better errors).
    strip_symbols: Whether to strip extra symbols not needed for execution
      (but which may aid debugging).
    crash_reproducer_path: File name to output an MLIR crash dump to if there
      is a compiler failure.
    enable_tflite_bindings: Support the IREE TFLite runtime bindings API shim.
    enable_benchmark: Whether to generate instrumented binaries suitable
      for benchmarking.
  """

  def __init__(self,
               *,
               output_file: Optional[str] = None,
               target_backends: Sequence[str] = (),
               output_format: Union[OutputFormat,
                                    str] = OutputFormat.FLATBUFFER_BINARY,
               extra_args: Sequence[str] = (),
               optimize: bool = True,
               output_mlir_debuginfo: bool = True,
               output_generic_mlir: bool = False,
               extended_diagnostics: bool = False,
               strip_debug_ops: bool = False,
               strip_source_map: bool = False,
               strip_symbols: bool = False,
               crash_reproducer_path: Optional[str] = None,
               enable_tflite_bindings: bool = False,
               enable_benchmark: bool = False):
    self.output_file = output_file
    self.target_backends = target_backends
    self.output_format = OutputFormat.parse(output_format)
    self.extra_args = extra_args
    self.optimize = optimize
    self.output_mlir_debuginfo = output_mlir_debuginfo
    self.output_generic_mlir = output_generic_mlir
    self.extended_diagnostics = extended_diagnostics
    self.strip_debug_ops = strip_debug_ops
    self.strip_source_map = strip_source_map
    self.strip_symbols = strip_symbols
    self.crash_reproducer_path = crash_reproducer_path
    self.enable_tflite_bindings = enable_tflite_bindings
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
  ]
  for target_backend in options.target_backends:
    cl.append(f"--iree-hal-target-backends={target_backend}")

  # Output file.
  if options.output_file:
    cl.append(f"-o={options.output_file}")

  # Translation to perform.
  cl.append("--iree-mlir-to-vm-bytecode-module")

  # MLIR flags.
  if options.output_mlir_debuginfo:
    cl.append("--mlir-print-debuginfo")
  if options.output_generic_mlir:
    cl.append("--mlir-print-op-generic")
  if options.extended_diagnostics:
    # Note that different tools have different defaults, so be explicit.
    cl.append("--mlir-print-op-on-diagnostic=true")
  else:
    cl.append("--mlir-print-op-on-diagnostic=false")

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
  if options.enable_tflite_bindings:
    cl.append("--iree-tflite-bindings-support")
  if options.enable_benchmark:
    cl.append("--iree-flow-export-benchmark-funcs")

  cl.extend(options.extra_args)
  return cl


def compile_file(input_file: str, **kwargs):
  """Invokes the IREE compiler on an input file.

  Args:
    input_file: File containing MLIR assembly to compile.
    **kwargs: Keyword arguments corresponding to CompilerOptions.
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


def compile_str(input_str: Union[str, bytes], **kwargs):
  """Invokes the IREE compiler with an input string.

  Args:
    input_str: MLIR assembly to parse/compile (str or bytes).
    **kwargs: Keyword arguments corresponding to CompilerOptions.
  Returns:
    Either a byte buffer of the compiled content or None if output_file
    was specified in the options.
  """
  options = CompilerOptions(**kwargs)
  cl = build_compile_command_line("-", options)
  input_bytes = input_str.encode("utf-8") if isinstance(input_str,
                                                        str) else input_str
  result = invoke_immediate(cl, immediate_input=input_bytes)
  if options.output_file:
    return None
  return result
