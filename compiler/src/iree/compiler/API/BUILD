# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

# Allow all python sources to be built externally.
exports_files(glob([
    "python/*.cpp",
    "python/*.c",
    "python/**/*.py",
    "unittests/**/*.py",
]))

################################################################################
# Package-by-package filegroup exports.
# Some tools can make use of such non-nested groups.
################################################################################

filegroup(
    name = "python_compiler_tools",
    srcs = glob(["python/iree/compiler/tools/*.py"]),
)

filegroup(
    name = "python_compiler_tools_deprecated_redirects",
    srcs = [
        "python/iree/compiler/__init__.py",
        "python/iree/compiler/tf.py",
        "python/iree/compiler/tflite.py",
        "python/iree/compiler/xla.py",
    ],
)

filegroup(
    name = "python_compiler_api",
    srcs = glob(["python/iree/compiler/transforms/*.py"]),
)

filegroup(
    name = "unittests_tools_testdata",
    srcs = glob([
        "python/test/tools/testdata/*",
    ]),
)
