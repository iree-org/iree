# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse, os

from library import *
from manifest import *
from launchers import *
from concurrent.futures import ThreadPoolExecutor
from options import parse_compile_arguments

###############################################################################
# Compile main : The main entry point for the compile tool.
# This tool compiles IREE-compiled MLIR operations for a given backend device,
###############################################################################

if __name__ == "__main__":
  ###############################################################################
  # Parse command line arguments
  ###############################################################################
  parser = argparse.ArgumentParser(
      description=
      "IREE Python compile tool for launching iree-compile for verification and "\
      "profiling. Issues iree-compile for a given backend device and iree-compile "\
      "flags. Uses ThreadPoolExecutor to launch multiple iree-compile processes "\
      "in parallel.")

  args = parse_compile_arguments(parser)
  ###############################################################################

  # Manifests metadata for a group of accompanying operations and configurations.
  manifest = Manifest(args)
  manifest.load()

  # Try and use all CPUs to launch iree-compile in parallel.
  cpu_count = os.cpu_count()
  if args.num_cpu > 0:
    cpu_count = min(cpu_count, args.num_cpu)

  # For all the operations in the manifest, issue iree-compile for verification
  # and profiling in parallel using ThreadPoolExecutor and cpu_count threads.
  cmds = []
  with ThreadPoolExecutor(max_workers=cpu_count) as executor:

    # For all the operations in the manifest compile, verify, and profile.
    for _, dispatch_collection_list in manifest.dispatch_collection_map.items():
      for dispatch_collection in dispatch_collection_list:
        # Create an instance of operation_launcher.
        operation = dispatch_collection.operation
        operation_launcher = IreeToolsLauncher(args, operation)
        for configuration in dispatch_collection.configuration_list:
          for compile_mode in [CompilationMode.Profile, CompilationMode.Verify]:
            cmds.append(executor.submit(\
              operation_launcher.iree_compile, compile_mode))

  # Wait for all the commands to complete.
  results = [cmd.result() for cmd in cmds]
