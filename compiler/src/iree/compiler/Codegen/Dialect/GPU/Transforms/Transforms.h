// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations for the IREE GPU dialect ------------===//
//
// Defines transformations that apply to IREE GPU ops for use in multiple
// places.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::vector {
struct UnrollVectorOptions;
}

namespace mlir::iree_compiler::IREE::GPU {

/// Function to fuse the given producer-consumer pair of forall loops into
/// the single consumer loop at the given |slice| within the consumer of the
/// producer. This is managed by inserting an `iree_gpu.tensor_shuffle` at the
/// boundary to synchronize the workers at the fusion point.
LogicalResult fuseForallIntoSlice(RewriterBase &rewriter,
                                  scf::ForallOp producer,
                                  scf::ForallOp consumer,
                                  tensor::ExtractSliceOp slice);

void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
