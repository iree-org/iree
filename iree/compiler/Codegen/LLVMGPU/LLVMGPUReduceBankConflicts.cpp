// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace iree_compiler {

/// Replace the uses of `oldOp` with the given `val` and for subview uses
/// propagate the type change. Changing the memref type may require propagating
/// it through subview ops so we cannot just do a replaceAllUse but need to
/// propagate the type change and erase old subview ops.
static void replaceUsesAndPropagateType(Operation *oldOp, Value val,
                                        OpBuilder &builder) {
  SmallVector<Operation *> opToDelete;
  SmallVector<OpOperand *> operandsToReplace;
  for (OpOperand &use : oldOp->getUses()) {
    auto subviewUse = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (!subviewUse) {
      // Save the operand to and replace outside the loop to not invalidate the
      // iterator.
      operandsToReplace.push_back(&use);
      continue;
    }
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
  }
  for (OpOperand *operand : operandsToReplace) operand->set(val);
  // Clean up old subview ops.
  for (Operation *op : opToDelete) op->erase();
}

/// Padd out the inner dimension of the allocOp in order reduce the chances to
/// have bank conflicts when reading 2D shapes within shared memory.
static void padAlloc(memref::AllocOp allocOp) {
  int64_t innerDim = allocOp.getType().getShape().back();
  if (ShapedType::isDynamic(innerDim)) return;
  Type elType = allocOp.getType().getElementType();
  unsigned bitwidth =
      mlir::DataLayout::closest(allocOp).getTypeSizeInBits(elType);
  // Pad with 128bits==16bytes so that accesses are still aligned on 16bytes.
  int64_t paddingSize = 128 / bitwidth;
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  shape.back() = shape.back() + paddingSize;
  MemRefType allocType = MemRefType::get(
      shape, elType, {}, allocOp.getType().getMemorySpaceAsInt());
  OpBuilder builder(allocOp);
  Location loc = allocOp.getLoc();
  Value paddedAlloc = builder.create<memref::AllocOp>(loc, allocType);
  SmallVector<int64_t> offsets(shape.size(), 0);
  SmallVector<int64_t> strides(shape.size(), 1);
  Value subview = builder.create<memref::SubViewOp>(
      loc, paddedAlloc, offsets, allocOp.getType().getShape(), strides);
  replaceUsesAndPropagateType(allocOp, subview, builder);
  allocOp->erase();
}

namespace {

/// Pass to reduce the number of bank conflicts when accessing shared memory in
/// a 2D manner. This is a simple version just padding allocation.
/// This doesn't fully remove bank conflicts and increase the shared memory
/// usage. In order to get better memory access patterns we should do shared
/// memory swizzling which requires more complex transformations. This pass can
/// be removed once the better solution is implemented.
struct LLVMGPUReduceBankConflictsPass
    : public LLVMGPUReduceBankConflictsBase<LLVMGPUReduceBankConflictsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<memref::AllocOp> sharedMemAllocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (allocOp.getType().getMemorySpaceAsInt() ==
          gpu::GPUDialect::getWorkgroupAddressSpace()) {
        sharedMemAllocs.push_back(allocOp);
      }
    });
    for (memref::AllocOp alloc : sharedMemAllocs) padAlloc(alloc);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUReduceSharedMemoryBankConflicts() {
  return std::make_unique<LLVMGPUReduceBankConflictsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
