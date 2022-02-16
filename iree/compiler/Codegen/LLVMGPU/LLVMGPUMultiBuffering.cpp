// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace iree_compiler {

// This code could live upstream, there is a review under progress and based on
// whether this is considered generic enough this will be moved there:
// https://reviews.llvm.org/D119406

/// Return true if the op fully overwrite the given `buffer` value.
static bool overrideBuffer(Operation *op, Value buffer) {
  if (isa<memref::CopyOp>(op) && cast<memref::CopyOp>(op).target() == buffer)
    return true;
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) return false;
  OpOperand *outputOperand = genericOp.getOutputOperand(0);
  if (outputOperand->get() == buffer &&
      genericOp.getTiedBlockArgument(outputOperand).use_empty())
    return true;
  return false;
}

/// Replace the uses of `oldOp` with the given `val` and for subview uses
/// propagate the type change. Changing the memref type may require propagating
/// it through subview ops so we cannot just do a replaceAllUse but need to
/// propagate the type change.
static void replaceUsesAndPropagateType(Operation *oldOp, Value val,
                                        OpBuilder &builder) {
  SmallVector<Operation *> opToDelete;
  SmallVector<OpOperand *> operandsToReplace;
  for (OpOperand &use : oldOp->getUses()) {
    auto subviewUse = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (subviewUse) {
      builder.setInsertionPoint(subviewUse);
      Type newType = memref::SubViewOp::inferRankReducedResultType(
          subviewUse.getType().getRank(), val.getType().cast<MemRefType>(),
          extractFromI64ArrayAttr(subviewUse.static_offsets()),
          extractFromI64ArrayAttr(subviewUse.static_sizes()),
          extractFromI64ArrayAttr(subviewUse.static_strides()));
      Value newSubview = builder.create<memref::SubViewOp>(
          subviewUse->getLoc(), newType.cast<MemRefType>(), val,
          subviewUse.getMixedOffsets(), subviewUse.getMixedSizes(),
          subviewUse.getMixedStrides());
      replaceUsesAndPropagateType(subviewUse, newSubview, builder);
      opToDelete.push_back(use.getOwner());
      continue;
    }
    // Save the operand to and replace outside the loop to not invalidate the
    // iterator.
    operandsToReplace.push_back(&use);
  }
  for (OpOperand *operand : operandsToReplace) operand->set(val);
  // Clean up old subview ops.
  for (Operation *op : opToDelete) op->erase();
}

/// Transformation to do multi-buffering/array expansion to remove dependencies
/// on the temporary allocation between consecutive loop iterations.
// This is not a pattern as it requires propagating the new memref type to its
// uses and requires updating subview ops.
static bool multiBuffering(memref::AllocOp allocOp, unsigned multiplier) {
  DominanceInfo dom(allocOp->getParentOp());
  scf::ForOp candidateLoop;
  for (Operation *user : allocOp->getUsers()) {
    auto parentLoop = user->getParentOfType<scf::ForOp>();
    if (!parentLoop) return false;
    /// Make sure there is no loop carried dependency on the allocation.
    if (!overrideBuffer(user, allocOp.getResult())) continue;
    // If the user doesn't dominate all the other users keep looking.
    if (llvm::any_of(allocOp->getUsers(), [&](Operation *otherUser) {
          return !dom.dominates(user, otherUser);
        }))
      continue;
    candidateLoop = parentLoop;
    break;
  }
  if (!candidateLoop) return false;
  SmallVector<int64_t, 4> newShape(1, multiplier);
  ArrayRef<int64_t> oldShape = allocOp.getType().getShape();
  newShape.append(oldShape.begin(), oldShape.end());
  auto newMemref = MemRefType::get(newShape, allocOp.getType().getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   allocOp.getType().getMemorySpace());
  OpBuilder builder(allocOp);
  Location loc = allocOp->getLoc();
  auto newAlloc = builder.create<memref::AllocOp>(loc, newMemref);
  builder.setInsertionPoint(candidateLoop.getBody(),
                            candidateLoop.getBody()->begin());
  // Calculate the iteration index = ((iv - inital_val) / step) % multiplier.
  AffineExpr induc = getAffineDimExpr(0, allocOp.getContext());
  AffineExpr init = getAffineDimExpr(1, allocOp.getContext());
  AffineExpr step = getAffineDimExpr(2, allocOp.getContext());
  AffineExpr expr = ((induc - init).floorDiv(step)) % multiplier;
  auto map = AffineMap::get(3, 0, expr);
  std::array<Value, 3> operands = {candidateLoop.getInductionVar(),
                                   candidateLoop.getLowerBound(),
                                   candidateLoop.getStep()};
  Value bufferIndex = builder.create<AffineApplyOp>(loc, map, operands);

  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(bufferIndex);
  offsets.append(oldShape.size(), builder.getIndexAttr(0));
  strides.append(oldShape.size() + 1, builder.getIndexAttr(1));
  sizes.push_back(builder.getIndexAttr(1));
  for (int64_t size : oldShape) sizes.push_back(builder.getIndexAttr(size));
  auto dstMemref =
      memref::SubViewOp::inferRankReducedResultType(
          allocOp.getType().getRank(), newMemref, offsets, sizes, strides)
          .cast<MemRefType>();
  Value subview = builder.create<memref::SubViewOp>(loc, dstMemref, newAlloc,
                                                    offsets, sizes, strides);
  replaceUsesAndPropagateType(allocOp, subview, builder);
  allocOp.erase();
  return true;
}

namespace {
struct LLVMGPUMultiBufferingPass
    : public LLVMGPUMultiBufferingBase<LLVMGPUMultiBufferingPass> {
  LLVMGPUMultiBufferingPass(unsigned numBuffers) : numBuffers(numBuffers) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<memref::AllocOp> allocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) { allocs.push_back(allocOp); });
    // Apply multi-buffering to all of them.
    for (memref::AllocOp alloc : allocs) {
      if (!multiBuffering(alloc, numBuffers))
        // Stop if any buffer cannot be multi buffered as pipelining will assume
        // this happened.
        return signalPassFailure();
    }
  }

 private:
  unsigned numBuffers;
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUMultiBuffering(
    unsigned numBuffers) {
  return std::make_unique<LLVMGPUMultiBufferingPass>(numBuffers);
}

}  // namespace iree_compiler
}  // namespace mlir
