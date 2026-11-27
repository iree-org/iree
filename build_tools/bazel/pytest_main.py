# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Entry point of iree_py_test targets: runs pytest with the given arguments."""

import os
import sys

import pytest

if __name__ == "__main__":
    args = [
        # The runfiles tree is not a place for pytest's cache.
        "-p",
        "no:cacheprovider",
        # No configuration file: pytest would otherwise search the directories
        # above the runfiles tree for one, which are outside the build.
        "-c",
        os.devnull,
        "--rootdir",
        os.getcwd(),
    ]
    # Bazel reads the results of the individual tests from this report.
    xml_output_file = os.environ.get("XML_OUTPUT_FILE")
    if xml_output_file:
        args.append(f"--junitxml={xml_output_file}")
    sys.exit(pytest.main(args + sys.argv[1:]))
