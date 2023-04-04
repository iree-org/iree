import argparse

from library import *
from matmul import *
from manifest import *
from performance_report import *
from options import parse_profiler_arguments

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

  args = parse_profiler_arguments(parser)
  ###############################################################################

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
        if args.compile_only:
          operation_launcher.compile(CompilationMode.Verify)
          operation_launcher.compile(CompilationMode.Profile)

        else:
          # Initialize verification and profiling results.
          verification_result = 'Not verified' if not args.verification_enabled else 'Failed'
          runtime = -1.0

          # Launch the operation dispatches for verification and profiling.
          if args.verification_enabled:
            verification_result = operation_launcher.verify(configuration)
          if args.profiling_enabled:
            runtime = operation_launcher.profile(configuration)

          # Save and print the performance result.
          if args.verification_enabled or args.profiling_enabled:
            # Create and print a performance result.
            result = PerformanceResult(operation_collection.operation,
                                       configuration, verification_result,
                                       runtime)
            result.print()

            # Append the performance result to the performance report.
            perf_report.append_perf_result(result)
