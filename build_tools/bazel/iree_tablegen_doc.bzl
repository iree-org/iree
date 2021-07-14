# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:tblgen.bzl", "gentbl_filegroup")

def iree_tablegen_doc(*args, **kwargs):
    """iree_tablegen_doc() generates documentation from a table definition file.

    This is a simple wrapper over gentbl() so we can differentiate between
    documentation and others. See gentbl() for details regarding arguments.
    """

    gentbl_filegroup(*args, **kwargs)
