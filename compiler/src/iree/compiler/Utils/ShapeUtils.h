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

/// Helper to compare shapes of two shaped types for equality.
///
/// For dimensions that are both dynamic, SSA value equality is required.
/// For dimensions that are both static, the sizes must match.
///
/// If `allowCasting` is false (default), a static dim will not match a dynamic
/// dim even if the dynamic value is a constant. This strict mode is suitable
/// for fold patterns where type equality matters.
///
/// If `allowCasting` is true, a static dim can match a dynamic dim if the
/// dynamic value is a constant with the same size.
bool compareShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                        ShapedType rhsType, ValueRange rhsDynamicDims,
                        bool allowCasting = false);

/// Same as compareShapesEqual, but ignores the last dimension.
bool compareShapesEqualExceptLastDim(ShapedType lhsType,
                                     ValueRange lhsDynamicDims,
                                     ShapedType rhsType,
                                     ValueRange rhsDynamicDims,
                                     bool allowCasting);

/// Helper to check whether 'from' is castable to the target ranked tensor type.
bool isCastableToTensorType(Type from, RankedTensorType to);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_SHAPEUTILS_H_
