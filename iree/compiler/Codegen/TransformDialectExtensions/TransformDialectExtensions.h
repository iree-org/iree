// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Registers transformations that require IREE-specific information into the
/// LinalgTransform dialect.
void registerLinalgTransformDialectExtension(DialectRegistry &registry);

//===----------------------------------------------------------------------===//
// Declaration of Patterns used in Transform Dialect extensions.
//==-----------------------------------------------------------------------===//

namespace IREE {
/// Pattern to rewrite a InParallelOp to the HAL dialect.
struct InParallelOpToHALRewriter
    : public OpRewritePattern<LinalgExt::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Operation *>> returningMatchAndRewrite(
      LinalgExt::InParallelOp inParallelOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(LinalgExt::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(inParallelOp, rewriter);
  }
};
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
