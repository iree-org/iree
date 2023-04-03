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
  parser.add_argument("--build-dir", default=".", \
                      help="IREE top-level build directory is used to generate "\
                      "operations and npy files.This should be same that used "\
                      "to call generated.py")

  parser.add_argument("--operation-kind", "--op-kind", dest="operation_kind",
                      default="all", help="Specifies the "\
                      "operation kinds to generate.", choices=["matmul", "conv2d", "all"])

  parser.add_argument("--verbose", default='False', \
                      help='Prints verbose output and commands executed.')

  parser.add_argument("--dispatches", default='', help="Comma delimited list to "\
                      "filter dispatches by name. A dispatch is a combination of "\
                      "operation and tuning configuration.")

  parser.add_argument("--mlir-dialect", default='linalg', help="MLIR dialect entry "\
                      "point at which operation is emitter.",
                      choices=["linalg", "flow", "all"])


def add_compilation_arguments(parser):
  """Adds compilation (NOT part of iree-compile) command line arguments to the parser."""

  compilation_parser = parser.add_argument_group(
      'Compilation', 'Compilation related options.')

  compilation_parser.add_argument("--force-compile", default='False', \
                      type=str, help="Force re-compilation of the operation even "\
                      "if .vmfb file is present.")
  compilation_parser.add_argument("--compile-only", default='False', \
                      type=str, help="Compiles the operation "\
                        "without running verification and profiling.")


def add_iree_compile_arguments(parser):
  """Adds iree-compile command line arguments to the parser."""
  iree_compile_parser = parser.add_argument_group(
      'iree-compile', 'iree-compile related options.')

  iree_compile_parser.add_argument("--iree-hal-target-backends", "--device", \
                      dest="device", default="cuda", \
                      help="Target backends for executable compilation. ", \
                      choices=["cuda", "vulkan", "cpu"])
  iree_compile_parser.add_argument("--iree-hal-cuda-llvm-target-arch", "--cuda-arch", \
                      dest="cuda_arch", default='sm_80', \
                      help="Target architecture for the CUDA backend. ", \
                      choices=["sm_50", "sm_60", "sm_75", "sm_80", "sm_86"])
  iree_compile_parser.add_argument('--iree-hal-benchmark-dispatch-repeat-count', '--batch-size',  \
                      dest="batch_size", default=100,
                      help="Number of times dispatch is launched in a loop to amortize the "\
                      "launch overhead. This argument is used for iree-compile and iree-benchamrk-module. "\
                      "The value used for iree-compile and iree-benchamrk-module should be the same.")


def add_verification_arguments(parser):
  """Adds verification related arguments to the parser."""
  verification_parser = parser.add_argument_group(
      'Verification', 'Verification related options.')

  verification_parser.add_argument("--verification-enabled", "--verify", default='True', \
                      type=str, help="Verify the operation.")
  verification_parser.add_argument("--verification-providers", default='numpy', \
                      choices=["numpy", "triton"], help="Comma delimited list of verification providers.")


def add_profiling_arguments(parser):
  """Adds profiling related arguments to the parser."""
  profiling_parser = parser.add_argument_group(
      'Profiling', 'Profiling (iree-benchmark-module) related options.')

  profiling_parser.add_argument("--profiling-enabled", "--benchmark", default='True', \
                      type=str, help="Benchmark the operation.")
  profiling_parser.add_argument("--benchmark-repetitions", default=5,
                      type=int, help="Number of times benchmark is repeated "\
                      "and min, max, median, and average runtimes/gflops are "\
                      "reported.")


def add_performance_report_arguments(parser):
  """Adds performance report related arguments to the parser."""

  performance_report_parser = parser.add_argument_group(
      'Performance Report', 'Performance report related options.')

  performance_report_parser.add_argument("--output", default='', \
                      help="Path to output file for csv readable results.")
  performance_report_parser.add_argument("--append", default='false', \
                      help="If true, result is appended to possibly existing file. "\
                        "Otherwise, any existing file is overwritten.")

  performance_report_parser.add_argument("--tags", default='', \
                      help="Inserts leading columns in output table and uniform "\
                        "values for each column. Useful for generating pivot tables.")


###############################################################################
# Create a parser and add all the arguments for a script function:
# add_generator_arguments() for generator.py
# add_profiler_arguments() for profiler.py
###############################################################################


def add_generator_arguments(parser):
  """Adds all the arguments for the *generator.py* script."""
  add_typical_arguments(parser)
  add_iree_compile_arguments(parser)


def add_profiler_arguments(parser):
  """Adds all the arguments for the *profiler.py* script."""
  add_typical_arguments(parser)
  add_compilation_arguments(parser)
  add_iree_compile_arguments(parser)
  add_verification_arguments(parser)
  add_profiling_arguments(parser)
  add_performance_report_arguments(parser)