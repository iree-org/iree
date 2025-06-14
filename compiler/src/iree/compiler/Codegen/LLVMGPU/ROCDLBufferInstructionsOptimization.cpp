// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

Value createI1And(Location loc, ArrayRef<Value> values, OpBuilder &builder) {
  Value base =
      builder.create<arith::IndexCastUIOp>(loc, builder.getI1Type(), values[0]);
  for (Value value : values.drop_front()) {
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
// Note we currently dont support cases where muliple masks are ANDed or ORed
// together to form the final mask to a read but such support can be added where
// we track a set of valid masks and add that an AND or OR of valid masks is
// valid

void simplifyMaskOps(RewriterBase &rewriter, vector::CreateMaskOp maskOp) {
  Location loc = maskOp.getLoc();

  SmallVector<vector::TransferReadOp> validReads;
  // First determine if the mask meets the criteria of being either all ones or
  // all empty.
  SmallVector<Value> ValuesToAnd;
  SmallVector<Value> maskIndices = maskOp.getOperands();
  ArrayRef<int64_t> maskShape = maskOp.getResult().getType().getShape();
  bool isValid = true;
  for (auto [idx, maskIndex] : llvm::enumerate(maskIndices)) {

    std::optional<int64_t> constantValue = getConstantIndex(maskIndex);
    if (constantValue) {
      if (maskShape[idx] != constantValue) {
        isValid = false;
        break;
      }
    } else {
      if (maskShape[idx] != 1) {
        isValid = false;
        break;
      }
      ValuesToAnd.push_back(maskIndex);
    }
  }
  // Bail out if the mask doesnt meet the criteria or
  // is statically all 1's in which case we dont need
  // to do anything.
  if (!isValid || ValuesToAnd.empty()) {
    return;
  }

  for (Operation *user : maskOp.getResult().getUsers()) {
    auto readOp = dyn_cast<vector::TransferReadOp>(user);
    // Only TransferReadOps are supported.
    if (!readOp)
      continue;

    auto sourceType = dyn_cast<MemRefType>(readOp.getBase().getType());
    // only supported for fat raw buffers.
    if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType))
      continue;

    SmallVector<bool> inBounds = readOp.getInBoundsValues();
    // Only supported for reads that are fully in_bounds.
    if (inBounds.size() != sourceType.getRank() ||
        llvm::any_of(inBounds, [](bool inBound) { return !inBound; })) {
      continue;
    }

    rewriter.setInsertionPoint(readOp);
    Value selectValue = createI1And(loc, ValuesToAnd, rewriter);
    auto constantValue = rewriter.create<vector::SplatOp>(
        loc, readOp.getVectorType(), readOp.getPadding());

    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        loc, readOp.getVectorType(), readOp.getBase(), readOp.getIndices(),
        readOp.getPadding(), ArrayRef<bool>{inBounds});
    auto selectOp = rewriter.create<arith::SelectOp>(loc, selectValue,
                                                     newReadOp, constantValue);
    rewriter.replaceAllUsesWith(readOp, selectOp);
  }
}

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    SmallVector<vector::CreateMaskOp> maskOps;
    funcOp.walk(
        [&](vector::CreateMaskOp maskOp) { maskOps.push_back(maskOp); });

    IRRewriter rewriter(context);
    for (vector::CreateMaskOp maskOp : maskOps) {
      simplifyMaskOps(rewriter, maskOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
