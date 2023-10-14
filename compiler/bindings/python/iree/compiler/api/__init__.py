# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler invocation API.

This package defines Python API wrappers around the IREE compiler C embedding
API. Refer to the C documentation in bindings/c/iree/compiler/embedding_api.h
for the most up to date information.

The objects in the C API are represented in Python as classes:

* `iree_compiler_error_t`: Never escapes (raised as exceptions).
* `iree_compiler_session_t`: `Session` class.
* `iree_compiler_invocation_t`: `Invocation` class.
* `iree_compiler_source_t`: `Source` class.
* `iree_compiler_output_t`: `Output` class.

In MLIR parlance, the `Session` wraps an MLIRContext with a set of flags for
configuring the compiler and context setup. `Invocation` wraps a Module in
the process of being compiled.
"""

from .ctypes_dl import (
    _initializeGlobalCL,
    Session,
    Invocation,
    Source,
    Output,
)
