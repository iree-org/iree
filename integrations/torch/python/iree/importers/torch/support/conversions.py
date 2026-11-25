# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable, Sequence

import torch

from iree.importers.torch.support.ir_imports import (
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    IntegerType,
    IrType,
)

# When emitting constants, we have to create native IREE types.
TORCH_DTYPE_TO_IREE_TYPE: dict[torch.dtype, Callable[[], IrType]] = {
    torch.float8_e4m3fnuz: lambda: Float8E4M3FNUZType.get(),
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.uint8: lambda: IntegerType.get_signless(8),
    torch.int8: lambda: IntegerType.get_signless(8),
    torch.int16: lambda: IntegerType.get_signless(16),
    torch.int32: lambda: IntegerType.get_signless(32),
    torch.int64: lambda: IntegerType.get_signless(64),
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.qint8: lambda: IntegerType.get_signless(8),
    torch.quint8: lambda: IntegerType.get_signless(8),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
    torch.complex128: lambda: ComplexType.get(F64Type.get()),
    torch.float8_e4m3fn: lambda: Float8E4M3FNType.get(),
    torch.float8_e4m3fnuz: lambda: Float8E4M3FNUZType.get(),
    torch.float8_e5m2: lambda: Float8E5M2Type.get(),
    torch.float8_e5m2fnuz: lambda: Float8E5M2FNUZType.get(),
}

TORCH_DTYPE_TO_IREE_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "i8",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
    torch.qint8: "i8",
    torch.quint8: "i8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
    torch.float8_e4m3fn: "f8E4M3FN",
    torch.float8_e4m3fnuz: "f8E4M3FNUZ",
    torch.float8_e5m2: "f8E5M2",
    torch.float8_e5m2fnuz: "f8E5M2FNUZ",
}


def torch_dtyped_shape_to_iree_format(
    shape_or_tensor: Sequence[int] | torch.Tensor,
    /,
    dtype: torch.dtype | None = None,
) -> str:
    """Example:
    shape = [1, 2, 3]
    dtype = torch.bfloat16
    Returns
    "1x2x3xbf16"
    """
    if isinstance(shape_or_tensor, torch.Tensor):
        dtype = dtype or shape_or_tensor.dtype
        return torch_dtyped_shape_to_iree_format(shape_or_tensor.shape, dtype)
    else:
        if dtype is None:
            raise ValueError(
                "dtype must be provided when shape is given instead of a tensor."
            )
    shape_str = "x".join([str(d) for d in shape_or_tensor])
    shape_dtype_delimiter = "x" if len(shape_or_tensor) > 0 else ""
    dtype_str = TORCH_DTYPE_TO_IREE_TYPE_ASM[dtype]
    return f"{shape_str}{shape_dtype_delimiter}{dtype_str}"
