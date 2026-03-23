// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-codegen-rocdl-buffer-instructions-optimization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLBUFFERINSTRUCTIONSOPTIMIZATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Simplify masked buffer reads/loads with broadcast mask.
//
// When a vector.transfer_read or vector.maskedload from a fat_raw_buffer has
// a mask that is vector.broadcast(%scalar_i1), replace with an unmasked
// read/load + arith.select. If the mask is always true, just return the
// unmasked read/load directly.
//===----------------------------------------------------------------------===//

/// Check if a value is a vector.broadcast of a scalar i1. If so, return
/// the scalar source.
static Value getBroadcastScalarI1(Value mask) {
  auto broadcastOp = mask.getDefiningOp<vector::BroadcastOp>();
  if (!broadcastOp) {
    return nullptr;
  }
  Value source = broadcastOp.getSource();
  if (!source.getType().isInteger(1)) {
    return nullptr;
  }
  return source;
}

/// Check if a scalar i1 value is a constant true.
static bool isConstantTrue(Value scalarI1) {
  return matchPattern(scalarI1, m_One());
}

/// Simplify a masked vector.transfer_read from a fat_raw_buffer.
static bool simplifyMaskedTransferRead(IRRewriter &rewriter,
                                       vector::TransferReadOp readOp) {
  // Must have a mask.
  Value mask = readOp.getMask();
  if (!mask) {
    return false;
  }

  // Mask must be vector.broadcast of scalar i1.
  Value scalarMask = getBroadcastScalarI1(mask);
  if (!scalarMask) {
    return false;
  }

  // Source must be fat_raw_buffer.
  auto sourceType = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
    return false;
  }

  // Must be fully in_bounds.
  SmallVector<bool> inBounds = readOp.getInBoundsValues();
  if (llvm::any_of(inBounds, [](bool b) { return !b; })) {
    return false;
  }

  Location loc = readOp.getLoc();
  rewriter.setInsertionPoint(readOp);

  // Create unmasked read, preserving the original permutation map.
  auto newReadOp = vector::TransferReadOp::create(
      rewriter, loc, readOp.getVectorType(), readOp.getBase(),
      readOp.getIndices(), readOp.getPadding(), readOp.getPermutationMap(),
      ArrayRef<bool>(inBounds));

  if (isConstantTrue(scalarMask)) {
    // Always-true: just use the unmasked read directly.
    rewriter.replaceOp(readOp, newReadOp);
  } else {
    // Conditional: select between the read and the padding.
    auto paddingBroadcast = vector::BroadcastOp::create(
        rewriter, loc, readOp.getVectorType(), readOp.getPadding());
    auto selectOp = arith::SelectOp::create(rewriter, loc, scalarMask,
                                            newReadOp, paddingBroadcast);
    rewriter.replaceOp(readOp, selectOp);
  }
  return true;
}

/// Simplify a masked vector.maskedload from a fat_raw_buffer.
static bool simplifyMaskedLoad(IRRewriter &rewriter,
                               vector::MaskedLoadOp maskedLoadOp) {
  // Mask must be vector.broadcast of scalar i1.
  Value scalarMask = getBroadcastScalarI1(maskedLoadOp.getMask());
  if (!scalarMask) {
    return false;
  }

  // Source must be fat_raw_buffer.
  auto sourceType = dyn_cast<MemRefType>(maskedLoadOp.getBase().getType());
  if (!sourceType || !hasAMDGPUFatRawBufferAddressSpace(sourceType)) {
    return false;
  }

  Location loc = maskedLoadOp.getLoc();
  rewriter.setInsertionPoint(maskedLoadOp);

  // Create unmasked vector.load.
  auto loadOp =
      vector::LoadOp::create(rewriter, loc, maskedLoadOp.getResult().getType(),
                             maskedLoadOp.getBase(), maskedLoadOp.getIndices());

  if (isConstantTrue(scalarMask)) {
    // Always-true: just use the unmasked load directly.
    rewriter.replaceOp(maskedLoadOp, loadOp);
  } else {
    // Conditional: select between the load and the passthru.
    auto selectOp = arith::SelectOp::create(rewriter, loc, scalarMask, loadOp,
                                            maskedLoadOp.getPassThru());
    rewriter.replaceOp(maskedLoadOp, selectOp);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ROCDLBufferInstructionsOptimizationPass final
    : impl::ROCDLBufferInstructionsOptimizationPassBase<
          ROCDLBufferInstructionsOptimizationPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IRRewriter rewriter(&getContext());

    // Simplify masked buffer reads/loads with broadcast mask.
    SmallVector<Operation *> maskedOps;
    funcOp.walk([&](Operation *op) {
      if (isa<vector::TransferReadOp, vector::MaskedLoadOp>(op)) {
        maskedOps.push_back(op);
      }
    });
    for (Operation *op : maskedOps) {
      if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
        simplifyMaskedTransferRead(rewriter, readOp);
      } else if (auto maskedLoadOp = dyn_cast<vector::MaskedLoadOp>(op)) {
        simplifyMaskedLoad(rewriter, maskedLoadOp);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
