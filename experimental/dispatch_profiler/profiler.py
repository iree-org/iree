# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

from library import *
from matmul import *
from batch_matmul import *
from manifest import *
from performance_report import *
from launchers import *
from options import parse_profiler_arguments

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
           "verifcation and performance profiling tool for IREE-compiled "\
           "MLIR operations.")

  args = parse_profiler_arguments(parser)
  ###############################################################################

  # Create manifest object and load dispatches.
  manifest = Manifest(args)
  manifest.load()

  # Performance report
  perf_report = PerformanceReport(args)

  # For all the operations in the manifest compile (if needed), verify, and profile.
  for _, dispatch_collection_list in manifest.dispatch_collection_map.items():
    for dispatch_collection in dispatch_collection_list:

      operation = dispatch_collection.operation
      # Select and create an instance of operation_launcher for the operation.
      operation_launcher = IreeToolsLauncher(args, operation)
      for configuration in dispatch_collection.configuration_list:

        # Create a dispatch object.
        dispatch = Dispatch(operation, configuration)

        # Skip the dispatch if filter returns false.
        if not manifest.is_enabled(dispatch):
          continue

        # If dry run is enabled, skip the dispatch.
        if args.dry_run:
          print(f'[Dry run] : {dispatch.name()}')
          continue

        # Initialize verification and profiling results.
        verification_result = 'Not verified' if not args.verification_enabled else 'Failed'
        runtime = -1.0

        # Launch the operation dispatches for verification and profiling.
        if args.verification_enabled:
          verification_result = operation_launcher.verify(configuration)
        if args.profiling_enabled:
          runtime = operation_launcher.profile(configuration)

        # Create performance result.
        result = PerformanceResult(operation, configuration,
                                   verification_result, runtime)

        # Print the performance result.
        result.print()

        # Append the performance result to the performance report.
        perf_report.append_perf_result(result)
