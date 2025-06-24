// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include <optional>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-folds-to-transpose-loads"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUFOLDTRANSPOSELOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

template <typename T>
struct SubgroupTransformPatternBase : public OpRewritePattern<T> {
  SubgroupTransformPatternBase(MLIRContext *context,
                               ArrayRef<int64_t> workgroupSize,
                               int64_t subgroupSize)
      : OpRewritePattern<T>(context), workgroupSize(workgroupSize),
        subgroupSize(subgroupSize) {}

protected:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
};

struct CombineVectorTransposeWithTransferReadOpPattern
    : SubgroupTransformPatternBase<vector::TransposeOp> {
  CombineVectorTransposeWithTransferReadOpPattern(
      MLIRContext *context, ArrayRef<int64_t> workgroupSize,
      int64_t subgroupSize)
      : SubgroupTransformPatternBase<vector::TransposeOp>(
            context, workgroupSize, subgroupSize) {}
  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        op,
        "Transpose ops are not supported on MMA types, use mma.load instead.");
  }
};

namespace {
struct GPUFoldTransposeLoadsPass final
    : impl::GPUFoldTransposeLoadsPassBase<GPUFoldTransposeLoadsPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        mlir::iree_compiler::getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      funcOp.emitOpError(
          "unimplemented: Distribution with dynamic workgroup size.");
      return signalPassFailure();
    }
    auto subgroupSize = mlir::iree_compiler::getSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp.emitOpError(
          "unimplemented: Distribution with dynamic subgroup size.");
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<CombineVectorTransposeWithTransferReadOpPattern>(
        context, *workgroupSize, *subgroupSize);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler
