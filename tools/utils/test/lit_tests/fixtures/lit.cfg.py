# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lit configuration for test fixtures."""

import os
import tempfile
from pathlib import Path

import lit.formats

config.name = "IREE Utils Test Fixtures"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]

# Route artifacts to temp directory to avoid polluting source tree.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)

# Pass through environment variables.
config.environment = os.environ.copy()

# Add build directory tools to PATH for integration tests.
# Try to locate the build directory relative to the repo root.
repo_root = Path(__file__).resolve().parents[5]  # Go up from fixtures to repo root.
build_dir = repo_root.parent / "iree-build"
if build_dir.exists():
    iree_tools = build_dir / "tools"
    filecheck_dir = build_dir / "llvm-project" / "bin"
    if iree_tools.exists() and filecheck_dir.exists():
        path_additions = f"{iree_tools}:{filecheck_dir}"
        if "PATH" in config.environment:
            config.environment[
                "PATH"
            ] = f"{path_additions}:{config.environment['PATH']}"
        else:
            config.environment["PATH"] = path_additions
