# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lit config for IREE."""

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by way of evaluating runlit.cfg.py from
# runlit.site.cfg.py which in turn is evaluated by lit.py.
# pylint: disable=undefined-variable

import os
import platform
import subprocess
import tempfile

import lit.formats

config.name = "IREE"
config.suffixes = [".mlir", ".txt"]
config.test_format = lit.formats.ShTest(
    execute_external=True, force_execute_external=True
)

# Forward all IREE environment variables, as well as some passthroughs.
# Note: env vars are case-insensitive on Windows, so check matches carefully.
#     https://stackoverflow.com/q/7797269
passthrough_env_vars = [
    # The Vulkan loader uses this
    "VK_ICD_FILENAMES",
    # WindowsLinkerTool uses these from vcvarsall
    "VCTOOLSINSTALLDIR",
    "UNIVERSALCRTSDKDIR",
    "UCRTVERSION",
]
config.environment.update(
    {
        k: v
        for k, v in os.environ.items()
        if k.startswith("IREE_") or k in passthrough_env_vars
    }
)

# On macOS, ensure the SDK sysroot is available for clang invoked in tests.
if platform.system() == "Darwin" and "SDKROOT" not in config.environment:
    try:
        sdkroot = subprocess.check_output(
            ["/usr/bin/xcrun", "--show-sdk-path"], text=True
        ).strip()
        if sdkroot:
            config.environment["SDKROOT"] = sdkroot
    except (OSError, subprocess.CalledProcessError):
        pass

# Bazel hands these over relative to the runfiles root, CMake absolute. Resolve
# so one RUN line serves both.
_test_srcdir = os.environ.get("TEST_SRCDIR", "")
for _key, _value in list(config.environment.items()):
    if not _key.endswith("_PLUGIN") or not _value or os.path.isabs(_value):
        continue
    for _root in (_test_srcdir, os.path.join(_test_srcdir, "_main"), os.getcwd()):
        _candidate = os.path.join(_root, _value)
        if os.path.exists(_candidate):
            config.environment[_key] = os.path.abspath(_candidate)
            break

if config.environment.get("IREE_TEST_DEPS_PLUGIN"):
    config.available_features.add("iree_dynamic_plugins")

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
