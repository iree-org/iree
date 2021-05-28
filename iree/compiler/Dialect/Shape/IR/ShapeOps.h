// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Populates conversion patterns that perform folding and canonicalization of
// shape ops. These patterns are intended to be used with the dialect conversion
// framework.
void populateFoldConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h.inc"

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_
