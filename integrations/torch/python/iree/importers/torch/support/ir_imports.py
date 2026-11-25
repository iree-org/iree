# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: F403, F401
"""Unifies all imports of iree.compiler.ir into one place."""

from iree.compiler.dialects import (
    util as util_d,
)
from iree.compiler.ir import (
    Attribute,
    BF16Type,
    # Types.
    ComplexType,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MLIRError,
    Module,
    Operation,
    RankedTensorType,
    ShapedType,
    StringAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
    Value,
)
from iree.compiler.ir import (
    Type as IrType,
)
from iree.compiler.passmanager import (
    PassManager,
)
