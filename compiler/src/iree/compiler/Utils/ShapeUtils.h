// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_SHAPEUTILS_H_
#define IREE_COMPILER_UTILS_SHAPEUTILS_H_

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler {

/// Helper to compare shapes of two shaped types by SSA equivalence.
/// Requires that all sizes are statically equal with no extra analysis.
/// Dynamic dimensions with soft or provable equality will still return false.
///
/// This is in contrast with cast compatible comparison which allows static and
/// dynamic sizes to compare positively.
bool compareShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                        ShapedType rhsType, ValueRange rhsDynamicDims);

/// Similar to compareShapesEqual, but allows a static dim to match a dynamic
/// dim if the dynamic value is a constant with the same size.
bool compareMixedShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                             ShapedType rhsType, ValueRange rhsDynamicDims);

/// Same as compareMixedShapesEqual, but ignores the last dimension.
bool compareMixedShapesEqualExceptLast(ShapedType lhsType,
                                       ValueRange lhsDynamicDims,
                                       ShapedType rhsType,
                                       ValueRange rhsDynamicDims);

/// Helper to check whether 'from' is castable to the target ranked tensor type.
bool isCastableToTensorType(Type from, RankedTensorType to);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_SHAPEUTILS_H_
