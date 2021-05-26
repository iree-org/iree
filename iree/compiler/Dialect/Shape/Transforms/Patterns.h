// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Sets up legality for shape calculation materialization conversions.
void setupMaterializeShapeCalculationsLegality(ConversionTarget &target);

// Populates patterns that will materialize shape calculations for any
// GetRankedShape and related ops.
void populateMaterializeShapeCalculationsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context);

// Sets up legality for shape calculation materialization conversions.
void setupShapeToStandardLegality(ConversionTarget &target);

// Populates patterns that will convert shape calculations into standard ops.
void populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_
