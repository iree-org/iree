# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

###############################################################################
#                    Options ohh! too main options
###############################################################################
# This file organizes the plenty of options that once can pass to the profiler
# tool scripts for generating, compiling, verifying, and profiling IREE-compiled
# MLIR operations.
#
# The options are organized into groups: typical, compilation, iree-compile,
# verification, profiling, performance-reporting. Note that there is a function
# of each group.
###############################################################################


def add_typical_arguments(parser):
  """Adds typical command line arguments to the parser."""
  parser.add_argument("--iree-bin-dir", default="./tools", \
                      help="Directory containing IREE binaries, "\
                      "e.g. iree-compile, iree-benchmark-module, "\
                      "iree-run-module")
  parser.add_argument("--generated-dir", default=".", \
                      help="The dispatch profiler scripts generate "\
                      "mlir dispatches, compiled vmfbs, and reference_chache "\
                      "containing golden npy files in the generated-dir")
  parser.add_argument("--operation-kind","--op-kind", \
                      dest="operation_kind", default="all", \
                      help="Specifies the operation kinds to generate.", \
                      choices=["matmul", "conv2d", "all"])
  parser.add_argument("--dispatches", default='',
                      help="Comma delimited list to filter dispatches by name. "\
                      "A dispatch is a combination of operation and tuning "\
                      "configuration.")
  parser.add_argument("--mlir-dialect", default='linalg', \
                      help="MLIR dialect entry point at which operation is emitter.",
                      choices=["linalg"])
  parser.add_argument("--verbose", action='store_true', \
                      help='Prints verbose output and commands executed.')
  parser.add_argument("--dry-run", action='store_true', \
                      help='Prints commands that will be executed without actually '\
                      'executing them.')
  parser.add_argument("--default-config", action='store_true',
                      help="Adds a dispatch without a pre-defined "\
                      "tuning configuration. This dispatch will use "\
                      "default configuration from KernelsConfig.cpp.")


def add_compilation_arguments(parser):
  """Adds compilation (not part of iree-compile) command line arguments to the parser."""
  compilation_parser = parser.add_argument_group(
      'Compilation', 'Compilation related options.')
  compilation_parser.add_argument("--num-cpu", "-j", \
                      dest="num_cpu", type=int, default=-1, \
                      help="Number of cpu threads to use for compilation.")
  compilation_parser.add_argument("--force-compile", action='store_true', \
                      help="Force re-compilation of the operation even "\
                      "if .vmfb file is present.")


def add_iree_compile_arguments(parser):
  """Adds iree-compile command line arguments to the parser."""
  iree_compile_parser = parser.add_argument_group(
      'iree-compile', 'iree-compile related options.')

  iree_compile_parser.add_argument(
                      "--iree-hal-target-backends", "--device", \
                      dest="device", default="cuda", \
                      help="Target backends for executable compilation. ", \
                      choices=["cuda", "vulkan", "cpu"])
  iree_compile_parser.add_argument(
                      "--iree-hal-cuda-llvm-target-arch", "--cuda-arch", \
                      dest="cuda_arch", default='sm_80', \
                      help="Target architecture for the CUDA backend. ", \
                      choices=["sm_50", "sm_60", "sm_75", "sm_80", "sm_86"])
  iree_compile_parser.add_argument(
                      '--iree-hal-benchmark-dispatch-repeat-count', '--batch-size', \
                      dest="batch_size", default=100,
                      help="Number of times dispatch is launched in a loop to "\
                      "amortize the launch overhead. This argument is used for "\
                      "iree-compile and iree-benchamrk-module. The value used by "\
                      "iree-compile and iree-benchamrk-module should be the same.")
  iree_compile_parser.add_argument(
                      '--iree-flow-split-matmul-reduction', '--split-k-slices', \
                      dest="split_k_slices", default="", \
                      help="Number of slices to split the reduction K-dimension.")
  iree_compile_parser.add_argument(
                     '--iree-codegen-llvmgpu-use-mma-sync', '--use-mma-sync', \
                      dest="use_mma_sync", action='store_true', \
                      help="Use mma.sync instructions.")
  iree_compile_parser.add_argument('--iree-codegen-llvmgpu-use-wmma', '--use-wmma', \
                      dest="use_wmma", action='store_true', \
                      help="Use wmma instructions.")
  iree_compile_parser.add_argument('--mlir-print-ir-after-all', '--print-ir-after-all', \
                      dest="mlir_print_ir_after_all", action='store_true', \
                      help="Prints IR after all transformations and dumps a "\
                      "file print_ir_after_*.mlir file.")


def add_verification_arguments(parser):
  """Adds verification related arguments to the parser."""
  verification_parser = parser.add_argument_group(
      'Verification', 'Verification related options.')

  verification_parser.add_argument(
                      "--verification-enabled", default='True', \
                      type=str, help="Verify the operation.")
  verification_parser.add_argument(
                     "--verification-providers", default='numpy', \
                      choices=["numpy"],
                      help="Comma delimited list of verification providers.")


def add_profiling_arguments(parser):
  """Adds profiling related arguments to the parser."""
  profiling_parser = parser.add_argument_group(
      'Profiling', 'Profiling (iree-benchmark-module) related options.')

  profiling_parser.add_argument(
                      "--profiling-enabled", "--benchmark", default='True', \
                      type=str, help="Benchmark the operation.")
  profiling_parser.add_argument(
                      "--benchmark-repetitions", default=5,
                      type=int, help="Number of times benchmark is repeated "\
                      "and min, max, median, and average runtimes/gflops are "\
                      "reported.")


def add_performance_report_arguments(parser):
  """Adds performance report related arguments to the parser."""

  performance_report_parser = parser.add_argument_group(
      'Performance Report', 'Performance report related options.')

  performance_report_parser.add_argument("--output", default='', \
                      help="Path to output file for csv readable results.")
  performance_report_parser.add_argument("--append", action='store_true', \
                      help="Appends the results to existing file. "\
                      "o.w., the existing file is overwritten.")
  performance_report_parser.add_argument("--tags", default='', \
                      help="Inserts leading columns in output table "\
                      "and uniform values for each column. Useful for "\
                      "generating pivot tables.")


def add_matmul_arguments(parser):
  """Adds matmul related arguments to the parser."""

  matmul_parser = parser.add_argument_group(
      'Matmul', 'Matrix-multiplication related options.')
  matmul_parser.add_argument("--problem-m", default='256', \
                      help="M dimension of the matrix. "\
                      "--problem-m==<value>,<value_start:value_end:increment>*")
  matmul_parser.add_argument("--problem-n", default='256', \
                      help="N dimension of the matrix."\
                      "--problem-n==<value>,<value_start:value_end:increment>*")
  matmul_parser.add_argument("--problem-k", default='256', \
                      help="K dimension of the matrix."\
                      "--problem-k==<value>,<value_start:value_end:increment>*")


###############################################################################
# Parser all the arguments for a script function:
# parse_generator_arguments() for generator.py
# parse_profiler_arguments() for profiler.py
###############################################################################


def parse_generator_arguments(parser):
  """Adds and parse all the arguments for the *generator.py* script."""
  add_typical_arguments(parser)
  add_matmul_arguments(parser)
  add_iree_compile_arguments(parser)
  args = parser.parse_args()
  return args


def parse_compile_arguments(parser):
  """Adds and parse all the arguments for the *compile.py* script."""
  add_typical_arguments(parser)
  add_compilation_arguments(parser)
  add_iree_compile_arguments(parser)
  args = parser.parse_args()
  return args


def parse_profiler_arguments(parser):
  """Adds and parse all the arguments for the *profiler.py* script."""
  add_typical_arguments(parser)
  add_compilation_arguments(parser)
  add_iree_compile_arguments(parser)
  add_verification_arguments(parser)
  add_profiling_arguments(parser)
  add_performance_report_arguments(parser)

  # Additional arguments for the profiler.
  parser.add_argument("--save-cmds", action='store_true', \
                      help='Saves commands and their output that are executed '\
                      'by the profiler in a file.')

  args = parser.parse_args()

  # Boolenize the string arguments from command line. For these args, it makes easier
  # to read and convey the meaning. The boolean arguments below are specified as:
  # `--argument=<true|false>`
  args.verification_enabled = False if args.verification_enabled in [
      'False', 'false', '0'
  ] else True

  args.profiling_enabled = False if args.profiling_enabled in [
      'False', 'false', '0'
  ] else True

  return args


###############################################################################
# Helper functions for parsing command line arguments.
###############################################################################
def get_cmd_line_argument_ranges(arg):
  """Returns a list of values generated by range of the form start:end:increment."""
  if not arg:
    return []
  if ':' not in arg:
    return [int(arg)]
  range_elements = arg.split(':')
  start = int(range_elements[0])
  end = int(range_elements[1])
  increment = int(range_elements[2]) if len(range_elements) == 3 else 1
  return range(start, end, increment)


def get_cmd_line_argument_list(arg):
  """Returns a list of values generated by comma delimited string."""
  values = arg.split(',')
  range_list = []
  for val in values:
    range_list += get_cmd_line_argument_ranges(val)
  return range_list
