# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lit config for IREE's clang-tidy extension tests."""

# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats

config.name = "IREEClangTidy"
config.suffixes = [".c", ".cc"]
config.test_format = lit.formats.ShTest(execute_external=True)

config.environment.update(
    {k: v for k, v in os.environ.items() if k.startswith("IREE_")}
)

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
