// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREDUCEBANKCONFLICTSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Pad out the inner dimension of the `memref.alloc` op in order reduce the
/// chances to have bank conflicts when reading 2D shapes within shared memory.
static void padAlloc(MLIRContext *context, memref::AllocOp allocOp,
                     unsigned paddingSizeBits) {
  // No padding requested - skip.
  if (paddingSizeBits == 0) {
    return;
  }

  auto allocOpShape = allocOp.getType().getShape();
  if (allocOpShape.empty()) {
    return;
  }
  int64_t innerDim = allocOpShape.back();
  if (ShapedType::isDynamic(innerDim)) {
    return;
  }

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
  SmallVector<int64_t> offsets(shape.size(), 0);
  SmallVector<int64_t> strides(shape.size(), 1);
  ArrayRef<int64_t> sizes = allocOp.getType().getShape();
  // Before performing any transformation, verify that we can propagate the new
  // type through the program. This could fail due to collapse_shape ops in the
  // use chain of the alloc.
  MemRefType resultType = memref::SubViewOp::inferRankReducedResultType(
      sizes, allocType, offsets, sizes, strides);
  if (failed(canReplaceMemrefUsesAndPropagateType(allocOp.getResult(),
                                                  resultType))) {
    return;
  }
  Value paddedAlloc = memref::AllocOp::create(rewriter, loc, allocType);
  Value subview = memref::SubViewOp::create(rewriter, loc, paddedAlloc, offsets,
                                            sizes, strides);
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

    MemRefType allocType = cast<MemRefType>(allocOp.getType());
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

/// Find the underlying memref.alloc for a value, looking through view-like ops.
static memref::AllocOp findUnderlyingAlloc(Value source) {
  while (source) {
    if (auto allocOp = source.getDefiningOp<memref::AllocOp>()) {
      return allocOp;
    }
    if (auto viewOp = source.getDefiningOp<ViewLikeOpInterface>()) {
      source = viewOp.getViewSource();
      continue;
    }
    break;
  }
  return nullptr;
}

/// Collect padding hints from BankConflictPaddingHintOp wrapping shared memory
/// allocations. Returns a map from alloc to the desired padding in bits.
/// If multiple hints map to the same alloc with conflicting padding values,
/// the padding is set to 0 (no padding applied).
static DenseMap<memref::AllocOp, unsigned>
collectPaddingHints(FunctionOpInterface funcOp) {
  DenseMap<memref::AllocOp, unsigned> allocPaddingMap;

  funcOp.walk([&](IREE::GPU::BankConflictPaddingHintOp hintOp) {
    Value source = hintOp.getOperand();
    memref::AllocOp allocOp = findUnderlyingAlloc(source);
    if (!allocOp || !hasSharedMemoryAddressSpace(allocOp.getType())) {
      return;
    }

    unsigned padding = hintOp.getPaddingBits();
    auto [it, inserted] = allocPaddingMap.try_emplace(allocOp, padding);
    if (!inserted && it->second != padding) {
      // Conflicting hints for the same alloc — disable padding.
      hintOp->emitWarning()
          << "conflicting bank conflict padding hints for " << *allocOp << ": "
          << it->second << " bits vs " << padding << " bits";
      it->second = 0;
    }
  });

  return allocPaddingMap;
}

/// Remove all BankConflictPaddingHintOp from the function, replacing their
/// uses with their operands.
static void removePaddingHints(FunctionOpInterface funcOp) {
  SmallVector<IREE::GPU::BankConflictPaddingHintOp> hintsToRemove;
  funcOp.walk([&](IREE::GPU::BankConflictPaddingHintOp hintOp) {
    hintsToRemove.push_back(hintOp);
  });

  for (auto hintOp : hintsToRemove) {
    hintOp.replaceAllUsesWith(hintOp.getOperand());
    hintOp.erase();
  }
}

static unsigned computeEffectiveExtraBytes(
    mlir::FunctionOpInterface funcOp,
    const DenseMap<memref::AllocOp, unsigned> &allocPaddingMap,
    unsigned fallbackPaddingBits) {
  unsigned totalExtra = 0;

  funcOp.walk([&](memref::AllocOp allocOp) {
    if (!hasSharedMemoryAddressSpace(allocOp.getType()) ||
        !allocOp.getType().hasStaticShape()) {
      return;
    }

    // Use hint padding if available, otherwise the fallback.
    unsigned paddingBits = fallbackPaddingBits;
    auto it = allocPaddingMap.find(allocOp);
    if (it != allocPaddingMap.end()) {
      paddingBits = it->second;
    }
    if (paddingBits == 0) {
      return;
    }

    MemRefType allocType = allocOp.getType();
    ArrayRef<int64_t> shape = allocType.getShape();
    if (shape.empty()) {
      return;
    }

    int64_t outerProduct = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
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

    // Allocs with BankConflictPaddingHintOp hints use the hinted padding.
    // Allocs without hints fall back to the paddingBits option.
    DenseMap<memref::AllocOp, unsigned> hintMap = collectPaddingHints(funcOp);
    unsigned effectiveExtraBytes =
        computeEffectiveExtraBytes(funcOp, hintMap, paddingBits);

    // TODO: Move this shared memory limit check to lowering config selection
    // so that padding decisions are made alongside tile size decisions, rather
    // than silently dropping padding when the budget is exceeded.
    if (static_cast<unsigned>(currentSharedMemUsage) + effectiveExtraBytes >
        sharedMemLimit) {
      // Skip the pass if an overflow would occur.
      return;
    }

    if (failed(reduceSharedMemoryBankConflicts(funcOp, paddingBits))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult reduceSharedMemoryBankConflicts(mlir::FunctionOpInterface funcOp,
                                              unsigned paddingSizeBits) {
  // Collect padding hints from BankConflictPaddingHintOp.
  DenseMap<memref::AllocOp, unsigned> allocPaddingMap =
      collectPaddingHints(funcOp);

  SmallVector<memref::AllocOp> sharedMemAllocs;
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (hasSharedMemoryAddressSpace(allocOp.getType()) &&
        allocOp.getType().hasStaticShape()) {
      sharedMemAllocs.push_back(allocOp);
    }
  });
  // Remove the hint ops now that we've consumed their padding values. This
  // must happen before padding so that the alloc users are the actual
  // subview/expand_shape ops that replaceMemrefUsesAndPropagateType knows how
  // to handle.
  removePaddingHints(funcOp);

  for (memref::AllocOp alloc : sharedMemAllocs) {
    unsigned effectivePadding = paddingSizeBits;
    auto it = allocPaddingMap.find(alloc);
    if (it != allocPaddingMap.end()) {
      effectivePadding = it->second;
    }
    padAlloc(funcOp->getContext(), alloc, effectivePadding);
  }

  return success();
}

} // namespace mlir::iree_compiler
