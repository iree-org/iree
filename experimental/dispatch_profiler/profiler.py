import argparse

from library import *
from matmul import *
from manifest import *
from performance_report import *

###############################################################################
# Map of operation kinds to their dispatch launchers.
operation_launcher_map = {
    OperationKind.Matmul: MatmulOperationLauncher,
}
###############################################################################

###############################################################################
# Profiler main : The main entry point for the profiler tool.
###############################################################################
# This tool compiles, verifies, and profiles IREE-compiled MLIR operations for
# a given backend device, compiler flags, and tuning configuration.
#
# The dispatch profiler tool is organized based on below defintions:
# Operation: A MLIR operation that is generated or consumed by the
#       dispatch_profiler. For example, linalg.matmul, linalg.conv2d, etc.
# Configuration: A set of compile parameters that are used by iree-compile the
#       to choose a compilation pipeline (e.g. LLVMGPUTensorCore,
#       LLVMGPUTensorCoreMmaSync, LLVGPUCPU, etc.), performance tuning parameters
#       (e.g. workgroup size, tile size etc.).
# Dispatch: A combination of an operation and a configuration is launched by the
#       dispatch profiler for verification and performance profiling. Note that
#       a dispatch is not a MLIR operation it is binary executable that is launched
#       by the profiler. Additionaly, the goal of the tool is to also profile the
#       performance of the fusions and a dispatch for fusion is a combination of
#       multiple operations glued together and compiled into a single dispatch.
###############################################################################

if __name__ == "__main__":
  ###############################################################################
  # Parse command line arguments
  ###############################################################################
  parser = argparse.ArgumentParser(description="IREE Python profiler tool for "\
                                   "verifcation and performance profiling tool "\
                                    "for IREE-compiled MLIR operations.")
  ###############################################################################

  # General profiler options
  parser.add_argument("--build-dir", default=".", \
                      help="IREE top-level build directory is used to generate "\
                        "operations and npy files.This should be same that used "\
                        "to call generated.py")
  parser.add_argument("--operation_kind", default="all", help="Specifies the "\
                      "operation kinds to generate.", choices=["matmul", "conv2d", "all"])
  parser.add_argument("--verbose", default='False', \
                      help='Prints verbose output and commands executed.')

  # Generator-specific options
  parser.add_argument("--dispatches", default='', help="Comma delimited list to "\
                      "filter dispatches by name. A dispatch is a combination of "\
                      "operation and tuning configuration.")
  parser.add_argument("--mlir-dialect", default='linalg', help="MLIR dialect entry "\
                      "point at which operation is emitter.",
                      choices=["linalg", "flow", "all"])
  # Compilation-specific options
  parser.add_argument("--device", default="cuda", \
                      help="Target backend device to benchmark the operation on. "\
                        "For example, cuda, vulkan, etc.")
  parser.add_argument("--force-compile", default='False', \
                      type=str, help="Force re-compilation of the operation even "\
                      "if .vmfb file is present.")
  parser.add_argument("--compile-only", default='False', \
                      type=str, help="Compiles the operation "\
                        "without running verification and profiling.")

  # Profiling-specific options
  parser.add_argument("--profiling-enabled", "--benchmark", default='True', \
                      type=str, help="Benchmark the operation.")
  parser.add_argument('--batch-size', '--benchmark-dispatch-repeat-count', \
                      default=100, help="Number of times dispatch is launched "\
                        "in a loop to amortize the launch overhead.")
  parser.add_argument("--benchmark-repetitions", default=5,
                      type=int, help="Number of times benchmark is repeated "\
                      "and min, max, median, and average runtimes/gflops are "\
                      "reported.")

  # Verification-specific options
  parser.add_argument("--verification-enabled", default='True',
                      type=str, help="Verify the operation against reference numpy "\
                      "implementation.")

  # Performance reporting options
  parser.add_argument("--output", default='', \
                      help="Path to output file for csv readable results.")
  parser.add_argument("--append", default='false', \
                      help="If true, result is appended to possibly existing file. "\
                        "Otherwise, any existing file is overwritten.")

  parser.add_argument("--tags", default='', \
                      help="Inserts leading columns in output table and uniform "\
                        "values for each column. Useful for generating pivot tables.")

  # Parse the command line arguments.
  args = parser.parse_args()
  ###############################################################################

  # Boolenize the string arguments from command line.
  verification_enabled = False if args.verification_enabled in [
      'False', 'false', '0'
  ] else True
  profiling_enabled = False if args.profiling_enabled in [
      'False', 'false', '0'
  ] else True
  compile_only = False if args.compile_only in ['False', 'false', '0'] else True
  # Overrite verification and profiling if compile_only is set.
  if compile_only:
    verification_enabled = False
    profiling_enabled = False

  # Manifests metadata for a group of accompanying operations and configurations.
  manifest = Manifest(args)

  # Load all the pre-defined dispatches in a manifest.
  manifest.load()

  # Performance report
  perf_report = PerformanceReport(args)

  # For all the operations in the manifest compile, verify, and profile.
  for operation_kind, operation_collection_list in manifest.operations.items():
    for operation_collection in operation_collection_list:

      # Select and create an instance of operation_launcher for the operation with operation_kind.
      # print(operation_collection.operation.name())
      operation_launcher = operation_launcher_map[operation_kind](
          args, operation_collection.operation)

      for configuration in operation_collection.configuration_list:

        # Compile the operation dispatches for verification and profiling.
        if compile_only:
          operation_launcher.compile(CompilationMode.Verify)
          operation_launcher.compile(CompilationMode.Profile)

        else:
          # Initialize verification and profiling results.
          verification_result = 'Not verified' if not verification_enabled else 'Failed'
          runtime = -1.0

          # Launch the operation dispatches for verification and profiling.
          if verification_enabled:
            verification_result = operation_launcher.verify(configuration)
          if profiling_enabled:
            runtime = operation_launcher.profile(configuration)

          # Save and print the performance result.
          if verification_enabled or profiling_enabled:
            # Create and print a performance result.
            result = PerformanceResult(operation_collection.operation,
                                       configuration, verification_result,
                                       runtime)
            result.print()

            # Append the performance result to the performance report.
            perf_report.append_perf_result(result)
