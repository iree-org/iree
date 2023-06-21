# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Disable PyType, it does not seem to like the specialization pattern used in
# MLIR.
# pytype: skip-file
try:
    from ..ir import *
    from ..dialects import pdl
    from ._ods_common import (
        extend_opview_class as _ods_extend_opview_class,
        segmented_accessor as _ods_segmented_accessor,
        equally_sized_accessor as _ods_equally_sized_accessor,
        get_default_loc_context as _ods_get_default_loc_context,
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
    )
    from typing import Optional, overload, Sequence, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e
BoolArg = Optional[Union[bool, BoolAttr]]
IntListArg = Optional[Union[Sequence[int], ArrayAttr]]
StringArg = Optional[Union[str, StringAttr]]


def _defaulted_ensure(f):
    def inner(value, default=None):
        assert value is not None or default is not None
        return f(default if value is None else value)

    return inner


@_defaulted_ensure
def _ensure_int_array_attr(value: IntListArg):
    i64 = IntegerType.get_signless(64)
    if isinstance(value, Sequence):
        return ArrayAttr.get([IntegerAttr.get(i64, i) for i in value])
    return value


@_defaulted_ensure
def _ensure_bool_attr(value: BoolArg):
    if isinstance(value, bool):
        return BoolAttr.get(value)
    return value


@_defaulted_ensure
def _ensure_string_attr(value: StringArg):
    if isinstance(value, str):
        return StringAttr.get(value)
    return value


class LowerToLLVMOp:
    """Specialization for the LowerToLLVMOp class."""

    def __init__(
        self,
        *,
        reassociate_fp_reductions: BoolArg = None,
        enable_index_optimizations: BoolArg = None,
        enable_arm_neon: BoolArg = None,
        enable_arm_sve: BoolArg = None,
        enable_amx: BoolArg = None,
        enable_x86vector: BoolArg = None,
        enable_async: BoolArg = None,
        loc=None,
        ip=None
    ):
        super().__init__(
            _ensure_bool_attr(reassociate_fp_reductions, False),
            _ensure_bool_attr(enable_index_optimizations, False),
            _ensure_bool_attr(enable_arm_neon, False),
            _ensure_bool_attr(enable_arm_sve, False),
            _ensure_bool_attr(enable_amx, False),
            _ensure_bool_attr(enable_x86vector, False),
            _ensure_bool_attr(enable_async, False),
            loc=loc,
            ip=ip,
        )
