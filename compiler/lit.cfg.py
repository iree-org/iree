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
import tempfile

import lit.formats

config.name = "IREE"
config.suffixes = [".mlir", ".txt"]
config.test_format = lit.formats.ShTest()

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

# Bazel passes test plugin paths relative to the runfiles root, CMake absolute.
# Each becomes a substitution: lit's shell does not expand $VAR.
_test_srcdir = os.environ.get("TEST_SRCDIR", "")


def _resolve_test_plugin(key, value):
    if os.path.isabs(value):
        return value
    for root in (_test_srcdir, os.path.join(_test_srcdir, "_main"), os.getcwd()):
        candidate = os.path.join(root, value)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    lit_config.fatal(f"{key}={value} does not resolve from the runfiles root")


for _key in sorted(config.environment, key=len, reverse=True):
    if _key.startswith("IREE_TEST_") and _key.endswith("_PLUGIN"):
        config.environment[_key] = _resolve_test_plugin(_key, config.environment[_key])
        config.substitutions.append(("%" + _key.lower(), config.environment[_key]))

# Only set when the build carries the loadable test plugins.
if config.environment.get("IREE_TEST_DEPS_PLUGIN"):
    config.available_features.add("iree_dynamic_plugins")

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
