// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

LogicalResult vectorizeGatherToTransferGather(RewriterBase &rewriter,
                                              Operation *op,
                                              ArrayRef<int64_t> vectorSizes,
                                              ArrayRef<bool> scalableVecDims);

void populateVectorTransferGatherLoweringPatterns(RewritePatternSet &patterns);

}; // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_TRANSFORMS_H_
