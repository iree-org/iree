// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

LogicalResult vectorizeGatherLikeGenericToTransferGather(
    RewriterBase &rewriter, linalg::GenericOp linalgOp,
    ArrayRef<int64_t> vectorSizes = {}, ArrayRef<bool> scalableVecDims = {},
    bool vectorizeNDExtract = false);

LogicalResult
vectorizeLinalgExtGatherToTransferGather(RewriterBase &rewriter,
                                         IREE::LinalgExt::GatherOp gatherOp);

void populateVectorTransferGatherLoweringPatterns(RewritePatternSet &patterns);

}; // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
