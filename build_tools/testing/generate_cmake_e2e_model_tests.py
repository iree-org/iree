#!/usr/bin/env python3
## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a CMake file to build the benchmark suites."""

import string
import sys
import pathlib
import argparse

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent / ".." / "python"))

import e2e_model_tests.cmake_generator


def main():
  cmake_rules = e2e_model_tests.cmake_generator.generate_rules(
      pathlib.PurePath("${ROOT_ARTIFACT_DIR}"))
  print("\n".join(cmake_rules))


if __name__ == "__main__":
  main()
