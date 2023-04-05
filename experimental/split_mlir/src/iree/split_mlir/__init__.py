# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._split_mlir import extract_operation_list
from .execution import execute_operation_list
from .iree_execution import IreeExecutor, execute_mlir_with_iree

__all__ = [
    "execute_operation_list", "execute_mlir_with_iree",
    "extract_operation_list", "IreeExecutor"
]
