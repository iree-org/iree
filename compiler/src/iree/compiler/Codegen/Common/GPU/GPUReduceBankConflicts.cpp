// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREDUCEBANKCONFLICTSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Check if AllocOp has a CollapseShapeOp user.
static bool hasCollapseShapeUser(memref::AllocOp allocOp) {
  SmallVector<Operation *> users(allocOp->getUsers());
  while (!users.empty()) {
    auto user = users.pop_back_val();
    if (isa<memref::CollapseShapeOp>(user)) {
      return true;
    }
    if (isa<ViewLikeOpInterface>(user)) {
      for (auto u : user->getUsers()) {
        users.push_back(u);
      }
    }
  }
  return false;
}

/// Pad out the inner dimension of the `memref.alloc` op in order reduce the
/// chances to have bank conflicts when reading 2D shapes within shared memory.
static void padAlloc(MLIRContext *context, memref::AllocOp allocOp,
                     unsigned paddingSizeBits) {
  auto allocOpShape = allocOp.getType().getShape();
  if (allocOpShape.empty())
    return;
  int64_t innerDim = allocOpShape.back();
  if (ShapedType::isDynamic(innerDim))
    return;

  // Return if we have CollapseShape op as an user as padding in that case is
  // unsupported.
  if (hasCollapseShapeUser(allocOp))
    return;

  Type elType = allocOp.getType().getElementType();
  unsigned bitwidth =
      mlir::DataLayout::closest(allocOp).getTypeSizeInBits(elType);
  // Pad with the specified amount. This should be >= bank size and <= widest
  // load size.
  int64_t paddingSize = paddingSizeBits / bitwidth;
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  shape.back() = shape.back() + paddingSize;
  MemRefType allocType =
      MemRefType::get(shape, elType, MemRefLayoutAttrInterface{},
                      allocOp.getType().getMemorySpace());
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(allocOp);
  Location loc = allocOp.getLoc();
  Value paddedAlloc = rewriter.create<memref::AllocOp>(loc, allocType);
  SmallVector<int64_t> offsets(shape.size(), 0);
  SmallVector<int64_t> strides(shape.size(), 1);
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, paddedAlloc, offsets, allocOp.getType().getShape(), strides);
  replaceMemrefUsesAndPropagateType(rewriter, loc, allocOp, subview);
  rewriter.eraseOp(allocOp);
}

static int64_t computeSharedMemoryUsage(mlir::FunctionOpInterface funcOp) {
  int64_t totalSharedMemory = 0;

  auto walkResult = funcOp.walk([&](memref::AllocOp allocOp) -> WalkResult {
    if (!hasSharedMemoryAddressSpace(allocOp.getType())) {
      return WalkResult::advance();
    }

    if (!allocOp.getType().hasStaticShape()) {
      return WalkResult::interrupt();
    }

    MemRefType allocType = llvm::cast<MemRefType>(allocOp.getType());
    unsigned byteWidth =
        allocType.getElementType().isIndex()
            ? 8 // IREE's default byteWidth for indexes
            : IREE::Util::getTypeBitWidth(allocType.getElementType()) / 8;

    int64_t numElements = 1;
    for (auto dimSize : allocType.getShape()) {
      numElements *= dimSize;
    }

    int64_t allocSizeBytes = byteWidth * numElements;
    if (allocOp.getAlignment()) {
      int64_t alignmentInBytes = *allocOp.getAlignment();
      allocSizeBytes =
          llvm::divideCeil(allocSizeBytes, alignmentInBytes) * alignmentInBytes;
    }

    totalSharedMemory += allocSizeBytes;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return ShapedType::kDynamic;
  }
  return totalSharedMemory;
}

static unsigned computeEffectiveExtraBytes(mlir::FunctionOpInterface funcOp,
                                           unsigned paddingBits) {
  unsigned totalExtra = 0;

  funcOp.walk([&](memref::AllocOp allocOp) {
    if (hasSharedMemoryAddressSpace(allocOp.getType()) &&
        allocOp.getType().hasStaticShape()) {
      MemRefType allocType = llvm::cast<MemRefType>(allocOp.getType());

      ArrayRef<int64_t> shape = allocType.getShape();
      if (shape.empty())
        return;

      int outerProduct = 1;
      for (std::size_t i = 0; i < shape.size() - 1; ++i) {
        outerProduct *= shape[i];
      }

      unsigned bitWidth = 64; // IREE's default bitWidth for indexes
      auto elemType = allocType.getElementType();
      if (!elemType.isIndex()) {
        bitWidth = IREE::Util::getTypeBitWidth(elemType);
      }
      unsigned elemSize = bitWidth / 8;
      unsigned extraElements = paddingBits / bitWidth;

      totalExtra += outerProduct * extraElements * elemSize;
    }
  });

  return totalExtra;
}

/// Pass to reduce the number of bank conflicts when accessing shared memory in
/// a 2D manner. This is a simple version just padding allocation.
/// This doesn't fully remove bank conflicts and increase the shared memory
/// usage. In order to get better memory access patterns we should do shared
/// memory swizzling which requires more complex transformations. This pass can
/// be removed once the better solution is implemented.
struct GPUReduceBankConflictsPass final
    : impl::GPUReduceBankConflictsPassBase<GPUReduceBankConflictsPass> {
  using GPUReduceBankConflictsPassBase::GPUReduceBankConflictsPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    unsigned sharedMemLimit =
        target ? target.getWgp().getMaxWorkgroupMemoryBytes() : 64 * 1024;

    int64_t currentSharedMemUsage = computeSharedMemoryUsage(funcOp);
    if (currentSharedMemUsage == ShapedType::kDynamic) {
      // The shape is dynamic, it may become static in later passes
      // if it becomes static, the padding will be applied.
      return;
    }

    unsigned effectiveExtraBytes =
        computeEffectiveExtraBytes(funcOp, paddingBits);

    if (static_cast<unsigned>(currentSharedMemUsage) + effectiveExtraBytes >
        sharedMemLimit) {
      // Skip the pass if an overflow would occur.
      return;
    }

    if (failed(reduceSharedMemoryBankConflicts(funcOp, paddingBits)))
      signalPassFailure();
  }
};

} // namespace

LogicalResult reduceSharedMemoryBankConflicts(mlir::FunctionOpInterface funcOp,
                                              unsigned paddingSize) {
  SmallVector<memref::AllocOp> sharedMemAllocs;
  // Collect all the alloc operations.
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (hasSharedMemoryAddressSpace(allocOp.getType()) &&
        allocOp.getType().hasStaticShape()) {
      sharedMemAllocs.push_back(allocOp);
    }
  });
  for (memref::AllocOp alloc : sharedMemAllocs)
    padAlloc(funcOp->getContext(), alloc, paddingSize);

  // In the current form this always succeeds.
  return success();
}

} // namespace mlir::iree_compiler
