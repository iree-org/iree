# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Live types during runtime of a procedure trace. User code will
# operate on instances of these.

from collections.abc import Sequence

import torch

from iree.importers.torch.support.procedural.base import (
    Intrinsic,
    IrTrace,
)
from iree.importers.torch.support.ir_imports import (
    IrType,
    RankedTensorType,
    Value,
)

###############################################################################
# Tensors and scalars
###############################################################################


class IrTensor(Intrinsic):
    """An intrinsic that represents a tensor value.

    Carries additional metadata needed to resolve dimensions and original
    PyTorch attributes.
    """

    __slots__ = [
        "ir_type",
        "dtype",
    ]

    def __init__(self, ir_type: IrType, dtype: torch.dtype):
        assert isinstance(dtype, torch.dtype)
        ranked_ir_type = RankedTensorType(ir_type)
        self.ir_type = ranked_ir_type
        self.dtype = dtype


class IrImmediateTensor(IrTensor):
    """Represents a Value in the IR under construction during procedural tracing."""

    __slots__ = [
        "_ir_value",
    ]

    def __init__(self, ir_value: Value, dtype: torch.dtype):
        super().__init__(ir_value.type, dtype)
        self._ir_value = ir_value

    def __repr__(self):
        return f"IrValueTensor(@{self.ir_value})"

    def resolve_ir_values(self, _: IrTrace) -> Sequence[Value]:
        return (self._ir_value,)
