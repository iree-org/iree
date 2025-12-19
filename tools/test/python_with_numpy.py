# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python interpreter wrapper that has numpy available.

Used by lit tests that need numpy but run under Bazel's hermetic Python.
This script acts as a Python interpreter: it takes a script path as the
first argument and executes it.
"""

import runpy
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        script = sys.argv[1]
        sys.argv = sys.argv[1:]  # Shift argv so script sees correct args
        runpy.run_path(script, run_name="__main__")
