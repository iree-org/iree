# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

def iree_tablegen_doc(includes = [], **kwargs):
    """iree_tablegen_doc() generates documentation from a table definition file.

    This is a simple wrapper over gentbl() so we can differentiate between
    documentation and others. See gentbl() for details regarding arguments.
    """

    gentbl_filegroup(includes = includes + [
        "/compiler/src",
    ], **kwargs)

def iree_gentbl_cc_library(includes = [], **kwargs):
    """IREE version of gentbl_cc_library which sets up includes properly."""

    gentbl_cc_library(includes = includes + [
        "/compiler/src",
    ], **kwargs)

def iree_td_library(includes = [], **kwargs):
    """IREE version of td_library."""

    td_library(includes = includes + [
        "/compiler/src",
    ], **kwargs)
