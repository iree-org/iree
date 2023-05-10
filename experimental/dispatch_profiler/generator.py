# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from library import *
from matmul import *
from manifest import *
from options import parse_generator_arguments

###############################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates MLIR operations for "\
                     "verification and profiling of IREE compiled dispatches.")

  args = parse_generator_arguments(parser)

  # Manifest dispatches for a group of accompanying operations and configurations.
  manifest = Manifest(args)

  # Load all the pre-defined dispatches in a manifest.
  manifest.initialize()

  # Emit the dispatches in MLIR source files.
  manifest.emit()
