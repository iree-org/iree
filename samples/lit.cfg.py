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
config.test_format = lit.formats.ShTest(
    execute_external=True, force_execute_external=True
)
# Forward all IREE environment variables
passthrough_env_vars = ["VK_ICD_FILENAMES"]
config.environment.update(
    {
        k: v
        for k, v in os.environ.items()
        if k.startswith("IREE_") or k in passthrough_env_vars
    }
)

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

if any(
    _key.endswith("_PLUGIN") and _value for _key, _value in config.environment.items()
):
    config.available_features.add("iree_dynamic_plugins")

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
