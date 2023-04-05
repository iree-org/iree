# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .types import OperationList


def extract_operation_list(mlir_file_path: str,
                           function_name: str) -> OperationList:
  ...
