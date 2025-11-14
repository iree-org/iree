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
# Build directory detection: IREE_BUILD_DIR env var, or common locations.
build_dir = None
if "IREE_BUILD_DIR" in os.environ:
    build_dir = Path(os.environ["IREE_BUILD_DIR"])
else:
    # Try common locations relative to repo root.
    repo_root = Path(__file__).resolve().parents[5]
    parent = repo_root.parent
    candidates = [
        repo_root / "build",  # In-tree: repo/build/
        parent / f"{repo_root.name}-build",  # Sibling: ../main-build/
        parent / "builds" / repo_root.name,  # Builds dir: ../builds/main/
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "tools").is_dir():
            build_dir = candidate
            break

if build_dir and build_dir.exists():
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
