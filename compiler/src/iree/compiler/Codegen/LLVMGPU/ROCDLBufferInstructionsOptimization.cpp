// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

Value createi1And(Location loc, ArrayRef<Value> values, OpBuilder &builder) {
  Value base =
      builder.create<arith::IndexCastUIOp>(loc, builder.getI1Type(), values[0]);
  for (auto value : values.drop_front()) {
    Value rhs =
        builder.create<arith::IndexCastUIOp>(loc, builder.getI1Type(), value);
    base = builder.create<arith::AndIOp>(loc, base, rhs);
  }
  return base;
}
// Determine if the mask vector is all ones or all zeros and if so, then replace
// the masked transferReadOp with transferReadOp with no mask for when the mask
// is all ones and the padding values when the mask is all zeros. The pattern
// thus does the following optimization.
// clang-format off
// mask = vector.create_mask %0, ..., %n, %c8 : vector<1x ... x1x8xi1>
// %read = vector.transfer_read %memref, %mask : memref<..., amdgpu.raw_fat_buffer>
// becomes
// %padding = arith.constant dense<0> : vector<1x ... x1x8xbf16>
// %read = vector.transfer_read %memref : memref<..., amdgpu.raw_fat_buffer> // no mask!
// %masked_read
//   = arith.select %0 && ... && %n ? %read : %padding : index,vector<1x ... x1x8xbf16>
// clang-format on
class SimplifyMaskVectorTransferRead
    : public OpRewritePattern<vector::CreateMaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    auto loc = maskOp.getLoc();
    auto readOp = dyn_cast<vector::TransferReadOp>(
        *(maskOp.getResult().getUsers().begin()));
    if (!readOp) {
      return failure();
    }

    auto sourceType = dyn_cast<MemRefType>(readOp.getSource().getType());
    if (!sourceType) {
      return failure();
    }
    // This optimization only works on fat raw buffers.
    if (!hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
      return failure();
    }

    SmallVector<bool> inBounds = readOp.getInBoundsValues();
    if (inBounds.size() != sourceType.getRank()) {
      return failure();
    }

    SmallVector<Value> maskIndices = maskOp.getOperands();
    ArrayRef<int64_t> maskShape = maskOp.getResult().getType().getShape();
    SmallVector<Value> ValuesToAnd;
    for (auto [idx, i] : llvm::enumerate(maskIndices)) {
      auto constantValue = getConstantIndex(i);
      // We only support the pattern for reads that are in-bounds.
      if (inBounds[idx] != true) {
        return failure();
      }
      if (constantValue) {
        // We bail-out if we can statically determine that a mask dim is not
        // full.
        if (maskShape[idx] != constantValue) {
          return failure();
        }
      } else {
        // The non-constant dimensions are only allowed to be unit to ensure
        // an all ones or an all zeros mask.
        if (maskShape[idx] != 1) {
          return failure();
        }
        ValuesToAnd.push_back(i);
      }
    }
    // There are no non-constant unit dimensions which means this is
    // a always one mask, such masks should be folded by vector
    // canonicalizations so we dont handle it.
    if (ValuesToAnd.size() == 0) {
      return failure();
    }
    auto selectValue = createi1And(loc, ValuesToAnd, rewriter);
    auto constantValue = rewriter.create<vector::SplatOp>(
        loc, readOp.getVectorType(), readOp.getPadding());

    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        loc, readOp.getVectorType(), readOp.getSource(), readOp.getIndices(),
        readOp.getPadding(), ArrayRef<bool>{inBounds});
    rewriter.replaceOpWithNewOp<arith::SelectOp>(readOp, selectValue, newReadOp,
                                                 constantValue);

    return success();
  }
};

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<SimplifyMaskVectorTransferRead>(funcOp.getContext());
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler
