# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence
from iree.compiler import ir
from iree.compiler.dialects import transform
from ._preprocessing_transform_ops_gen import *
from ._preprocessing_transform_ops_gen import _Dialect

try:
    from ._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class MatchContractionOp(MatchContractionOp):
    def __init__(
        self,
        operand_handle,
        lhs_type: ir.Type,
        rhs_type: ir.Type,
        output_type: ir.Type,
        indexing_maps: Optional[Sequence] = None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = ir.Location.unknown()

        param_type = transform.ParamType.get(ir.IntegerType.get_signless(64))
        lhs_type_attr = ir.TypeAttr.get(lhs_type)
        rhs_type_attr = ir.TypeAttr.get(rhs_type)
        output_type_attr = ir.TypeAttr.get(output_type)

        indexing_maps_attr = None
        if indexing_maps is not None:
            indexing_maps_attr = ir.ArrayAttr.get(
                [ir.AffineMapAttr.get(m) for m in indexing_maps]
            )

        # Call auto-generated base constructor.
        super().__init__(
            param_type,
            param_type,
            param_type,
            param_type,
            operand_handle,
            lhs_type_attr,
            rhs_type_attr,
            output_type_attr,
            indexing_maps=indexing_maps_attr,
            loc=loc,
            ip=ip,
        )

    def __iter__(self):
        return iter([self.batch_dims, self.m_dims, self.n_dims, self.k_dims])


__all__ = [
    "MatchContractionOp",
]
