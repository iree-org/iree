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
import sys
import tempfile

import lit.formats

config.name = "IREE"
config.suffixes = [".mlir", ".txt"]
config.test_format = lit.formats.ShTest(execute_external=True)
# Forward all IREE environment variables
passthrough_env_vars = ["VK_ICD_FILENAMES"]
config.environment.update({
    k: v
    for k, v in os.environ.items()
    if k.startswith("IREE_") or k in passthrough_env_vars
})

# Use the most preferred temp directory.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))

config.substitutions.extend([
    ("%PYTHON", os.getenv("PYTHON", sys.executable)),
])
