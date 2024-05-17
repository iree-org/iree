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

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::linalg {
class LinalgOp;
}

namespace mlir::scf {
class ForallOp;
}

namespace mlir::vector {
struct UnrollVectorOptions;
}

namespace mlir::iree_compiler::IREE::GPU {

// Helper to vectorize a static multi-mma op.
LogicalResult vectorizeStaticMultiMmaOp(RewriterBase &rewriter,
                                        IREE::GPU::MultiMmaOp mmaOp);

// Helper to convert a contraction-like linalg op to an iree_gpu.multi_mma.
FailureOr<IREE::GPU::MultiMmaOp>
convertContractionToMultiMma(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             IREE::GPU::MmaInterfaceAttr mmaKind);

// Helper to distribute a multi_mma op to lanes.
FailureOr<Operation *> distributeMultiMmaOp(RewriterBase &rewriter,
                                            IREE::GPU::MultiMmaOp mmaOp);

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns);

void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);

void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns);

void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerVectorBarrierPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
