// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <optional>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-folds-to-transpose-loads"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUFOLDTRANSPOSELOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

struct FoldTransposeAndLoadPattern
    : public OpRewritePattern<vector::TransposeOp> {

  FoldTransposeAndLoadPattern(MLIRContext *context,
                              ArrayRef<int64_t> workgroupSize,
                              int64_t subgroupSize)
      : OpRewritePattern<vector::TransposeOp>(context),
        workgroupSize(workgroupSize), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    LDBG("==== folding transpose and load: ");

    return llvm::failure();
  }

  LogicalResult
  isTransferReadTransposeChain(vector::TransposeOp transposeOp) const {
    Value sourceVector = transposeOp.getVector();
    if (!isa<vector::TransferReadOp>(sourceVector.getDefiningOp())) {
      return failure();
    }
    if (!sourceVector.hasOneUse()) {
      return failure();
    }
    return success();
  }

  // TODO: mfma vectors have more than 2 dimensions.
  LogicalResult
  isColMajorToRowMajorTranspose(vector::TransposeOp transposeOp) const {
    auto perm = transposeOp.getPermutation();
    if (perm.size() != 2) {
      return failure();
    }
    // Check if the transpose is from column-major to row-major.
    if (!(perm[0] == 1 && perm[1] == 0)) {
      return failure();
    }
    return success();
  }

  LogicalResult isInsideValidForallOp(vector::TransposeOp transposeOp) const {
    auto parentOp = transposeOp->getParentOfType<scf::ForallOp>();
    if (!parentOp) {
      return failure();
    }
    // get the loop count from the parent forall op.
    auto lb = parentOp.getStaticLowerBound();
    auto ub = parentOp.getStaticUpperBound();
    auto step = parentOp.getStaticStep();

    // TODO: enable multiple dimension scf.forall ops.
    if (lb.size() != 1) {
      return failure();
    }
    if (step.front() != 1) {
      return failure();
    }
    if (lb.front() != 0) {
      return failure();
    }

    size_t loopCount = ub.front() - lb.front();

    // For now, loop count should be a multiple of the subgroup size.
    if (loopCount % subgroupSize != 0) {
      return failure();
    }

    return success();
  }

protected:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
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
    patterns.add<FoldTransposeAndLoadPattern>(context, *workgroupSize,
                                              *subgroupSize);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler
