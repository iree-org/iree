# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, TypeVar, Callable, Tuple
from numbers import Integral

Tensor = TypeVar("Tensor")
OpId = TypeVar("OpId")
ExecuteOp = Callable[[OpId, List[Tensor]], List[Tensor]]
OperationIndex = Integral
ResultIndex = Integral
"""Description of the dependencies of an operation."""
OpArguments = List[Tuple[OperationIndex, ResultIndex]]
Operation = Tuple[OpId, OpArguments]
"""Describes a dependency graph of operations."""
OperationList = List[Operation]
