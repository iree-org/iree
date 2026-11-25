# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from collections.abc import Sequence

import pytest
import torch
from iree.importers.torch.support.conversions import torch_dtyped_shape_to_iree_format


@pytest.mark.parametrize(
    "shape_or_tensor,dtype,expected",
    [
        [[1, 2, 3], torch.bfloat16, "1x2x3xbf16"],
        # From tensor
        [torch.empty([4, 5], dtype=torch.float32), None, "4x5xf32"],
        # Zero-rank shape
        [[], torch.float8_e4m3fn, "f8E4M3FN"],
        # Tuple as shape
        [(6,), torch.int8, "6xi8"],
        # torch.Size as shape
        [torch.Size([7, 8]), torch.quint8, "7x8xi8"],
    ],
)
def test_torch_dtyped_shape_to_iree_format(
    shape_or_tensor: Sequence[int] | torch.Tensor, dtype: torch.dtype, expected: str
):
    iree_format = torch_dtyped_shape_to_iree_format(shape_or_tensor, dtype)
    assert iree_format == expected


def test_torch_dtyped_shape_to_iree_format_missing_dtype():
    with pytest.raises(ValueError):
        torch_dtyped_shape_to_iree_format([1, 2], None)


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)
