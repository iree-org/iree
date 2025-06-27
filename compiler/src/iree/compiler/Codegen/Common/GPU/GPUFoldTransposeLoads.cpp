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
#define INDENT_LDBG(X) LLVM_DEBUG(llvm::dbgs() << "    " << X << "\n")

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
    LDBG(*transposeOp);
    if (failed(isTransferReadTransposeChain(transposeOp)) ||
        failed(isValidTranspose(transposeOp)) ||
        failed(isInsideValidForallOp(transposeOp))) {
      return failure();
    }

    return llvm::failure();
  }

  LogicalResult
  isTransferReadTransposeChain(vector::TransposeOp transposeOp) const {
    Value sourceVector = transposeOp.getVector();
    if (!isa<vector::TransferReadOp>(sourceVector.getDefiningOp())) {
      INDENT_LDBG("Skip: source is not a vector.transfer_read.");
      return failure();
    }
    if (!sourceVector.hasOneUse()) {
      INDENT_LDBG("Skip: source vector has more than one use.");
      return failure();
    }
    return success();
  }

  // TODO: mfma vectors have more than 2 dimensions.
  LogicalResult isValidTranspose(vector::TransposeOp transposeOp) const {
    auto perm = transposeOp.getPermutation();
    if (perm.size() != 2) {
      INDENT_LDBG("Skip: transpose permutation size is not 2.");
      return failure();
    }

    // Transpose should be from column-major to row-major.
    if (!(perm[0] == 1 && perm[1] == 0)) {
      INDENT_LDBG(
          "Skip: transpose permutation is not column-major to row-major.");
      return failure();
    }

    VectorType vectorType = transposeOp.getVector().getType();
    if (vectorType.getRank() != 2) {
      INDENT_LDBG("Skip: transpose vector is not 2D.");
      return failure();
    }

    size_t vectorElementBitWidth = vectorType.getElementTypeBitWidth();
    size_t vectorSizeInBits =
        vectorElementBitWidth * vectorType.getNumElements();

    switch (vectorSizeInBits) {
    case 64: {
      if (vectorElementBitWidth != 4 && vectorElementBitWidth != 8 &&
          vectorElementBitWidth != 16) {
        INDENT_LDBG("Skip: element bitwidth not legal.");
        return failure();
      }
      break;
    }
    case 96:
      if (vectorElementBitWidth != 6) {
        INDENT_LDBG("Skip: element bitwidth not legal.");
        return failure();
      }
      break;
    default:
      INDENT_LDBG("Skip: vector size in bits not legal: " << vectorSizeInBits);
      return failure();
      break;
    }

    return success();
  }

  LogicalResult isInsideValidForallOp(vector::TransposeOp transposeOp) const {
    auto parentOp = transposeOp->getParentOfType<scf::ForallOp>();
    if (!parentOp) {
      INDENT_LDBG("Skip: transpose is not inside a scf.forall op.");
      return failure();
    }
    // get the loop count from the parent forall op.
    auto lb = parentOp.getStaticLowerBound();
    auto ub = parentOp.getStaticUpperBound();
    auto step = parentOp.getStaticStep();

    // TODO: enable multiple dimension scf.forall ops.
    if (lb.size() != 1) {
      INDENT_LDBG("Skip: transpose is not inside a 1D scf.forall op.");
      return failure();
    }
    if (step.front() != 1) {
      INDENT_LDBG("Skip: transpose is not inside a scf.forall op with step 1.");
      return failure();
    }
    if (lb.front() != 0) {
      INDENT_LDBG("Skip: transpose is not inside a scf.forall op with lower "
                  "bound 0.");
      return failure();
    }

    INDENT_LDBG("scf.forall loop count: " << lb.front() << " to " << ub.front()
                                          << " with step " << step.front());
    size_t loopCount = ub.front() - lb.front();

    // For now, loop count should be a multiple of the subgroup size.
    if (loopCount % subgroupSize != 0) {
      INDENT_LDBG("Skip: surrounding scf.forall op looop count is not multiple "
                  "of subgroup size.");
      return failure();
    }

    // Is mutiple of recommended transpose dimension size.

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
