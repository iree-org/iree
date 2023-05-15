# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import tempfile

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

llvm_config.with_system_environment("PYTHONPATH")
llvm_config.with_system_environment("VK_ICD_FILENAMES")

# Put execution artifacts in the temp dir.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))

# name: The name of this test suite.
config.name = "TENSORFLOW_TESTS"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".run"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py",
    "test_util.py",
    "manual_test.py",
    "squad_test_data.py",
    "imagenet_test_data.py",
]

config.substitutions.extend([
    ("%PYTHON", os.getenv("PYTHON", sys.executable)),
])

# Add our local projects to the PYTHONPATH
python_projects_dir = os.path.join(os.path.dirname(__file__), "..",
                                   "python_projects")
test_src_dir = os.path.join(os.path.dirname(__file__), "python")
llvm_config.with_environment("PYTHONPATH", [
    test_src_dir,
    os.path.join(python_projects_dir, "iree_tf"),
    os.path.join(python_projects_dir, "iree_tflite"),
],
                             append_path=True)

# Enable features based on -D FEATURES=hugetest,vulkan
# syntax.
# We always allow "llvmcpu". It can be disabled with -D DISABLE_FEATURES=llvmcpu
disable_features_param = lit_config.params.get("DISABLE_FEATURES")
disable_features = []
if disable_features_param:
  disable_features = disable_features_param.split(",")
if "llvmcpu" not in disable_features:
  config.available_features.add("llvmcpu")
features_param = lit_config.params.get("FEATURES")
if features_param:
  config.available_features.update(features_param.split(","))
