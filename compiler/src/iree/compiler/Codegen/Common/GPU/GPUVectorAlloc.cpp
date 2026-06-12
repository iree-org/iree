// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUNestedLayoutUtils.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPromotionAnalysis.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

using LayoutMap =
    llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface>;

/// Allocate a workgroup-addressed tensor with the given shape and element type.
static bufferization::AllocTensorOp
allocateWorkgroupTensor(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                        Type elementType) {
  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(shape, elementType, sharedMemoryAddrSpace);
  auto allocTensorOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);
  return allocTensorOp;
}

static FailureOr<Value> writeVectorToTensor(OpBuilder &b, Location loc,
                                            Value vector, Value dest) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  llvm::Repeated<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied =
      vector::TransferWriteOp::create(b, loc, vector, dest, indices, inBounds)
          .getResult();
  return copied;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = arith::ConstantIndexOp::create(b, tensor.getLoc(), 0);
  llvm::Repeated<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return vector::TransferReadOp::create(b, tensor.getLoc(), vectorType, tensor,
                                        indices, /*padding=*/std::nullopt,
                                        inBounds)
      .getResult();
}

static void insertWorkgroupBarrierAtBlockStart(OpBuilder &builder,
                                               Operation *op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(op->getBlock());
  gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);
}

static LogicalResult
validateSharedMemoryConversionAttrs(FunctionOpInterface funcOp) {
  WalkResult result = funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    Attribute attr = op->getAttr("shared_memory_conversion");
    if (!attr) {
      return WalkResult::advance();
    }
    if (llvm::isa<IREE::GPU::PromotionAttr>(attr)) {
      return WalkResult::advance();
    }
    op.emitOpError("shared_memory_conversion attribute must implement "
                   "IREE::GPU::PromotionAttr");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

/// Materialize shared memory roundtrip to resolve layout differences. Clears
/// the shared_memory_conversion attribute after materialization.
static LogicalResult
materializeSharedMemoryConversions(FunctionOpInterface funcOp) {
  SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
  funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    if (op.getSharedMemoryConversion()) {
      opsToPromote.push_back(op);
    }
  });

  OpBuilder builder(funcOp);
  for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
    if (!llvm::isa<IREE::GPU::PromotionAttr>(
            op.getSharedMemoryConversionAttr())) {
      op.emitOpError("shared_memory_conversion attribute must implement "
                     "IREE::GPU::PromotionAttr");
      return failure();
    }

    // HACK: Until proper barrier placement is handled later we have to
    // synchronize explicitly in this pass.

    // Synchronize before the write to shared memory to avoid stepping over
    // reads in the previous iteration of a loop. We set this barrier
    // at the start of this block.
    insertWorkgroupBarrierAtBlockStart(builder, op);

    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();
    // TODO: Since we know the read/write layout for this memory, we can get
    // optimal swizzling here. Figure out how to do that.
    VectorType vectorType = cast<VectorType>(operand.get().getType());
    auto allocTensorOp =
        allocateWorkgroupTensor(builder, op->getLoc(), vectorType.getShape(),
                                vectorType.getElementType());
    FailureOr<Value> ret = writeVectorToTensor(builder, op->getLoc(),
                                               operand.get(), allocTensorOp);
    if (failed(ret)) {
      return failure();
    }

    // Synchronize after the write to shared memory before we read from it.
    auto synced =
        IREE::GPU::ValueBarrierOp::create(builder, op->getLoc(), *ret);

    VectorType inputTy = cast<VectorType>(op.getType());
    Value read = readVectorFromTensor(builder, inputTy, synced.getResult(0));
    operand.set(read);

    // Remove the shared_memory_conversion attribute from the to_layout
    // operation.
    op.removeSharedMemoryConversionAttr();
  }
  return success();
}

struct PromotionCandidate {
  Operation *op = nullptr;
  Value vector;
  Attribute promotionType;
};

template <typename... PromotionTypes>
static std::optional<PromotionCandidate>
shouldPromoteCandidate(const PromotionTypeMap &promotionTypes, Operation *op,
                       Value vector) {
  auto promotionType = promotionTypes.find(vector);
  if (promotionType == promotionTypes.end()) {
    return std::nullopt;
  }

  if (!isa<PromotionTypes...>(promotionType->second)) {
    return std::nullopt;
  }
  return PromotionCandidate{op, vector, promotionType->second};
}

/// Find promotion candidates, i.e., operations that load data from memory and
/// were reached by a promotion type during analysis propagation.
static std::optional<PromotionCandidate>
getPromotionCandidate(Operation *op, const PromotionTypeMap &promotionTypes) {
  return TypeSwitch<Operation *, std::optional<PromotionCandidate>>(op)
      .Case([&](vector::TransferReadOp read) {
        return shouldPromoteCandidate<IREE::GPU::DerivedThreadConfigAttr,
                                      IREE::GPU::UseGlobalLoadDMAAttr>(
            promotionTypes, read.getOperation(), read.getVector());
      })
      .Case([&](vector::GatherOp gather) {
        // TODO: Currently, we support use_global_load_dma only for read ops.
        return shouldPromoteCandidate<IREE::GPU::DerivedThreadConfigAttr>(
            promotionTypes, gather.getOperation(), gather.getResult());
      })
      .Default(std::nullopt);
}

static bool hasAllInBounds(vector::TransferReadOp readOp) {
  std::optional<ArrayAttr> inBounds = readOp.getInBounds();
  return inBounds && llvm::all_of(*inBounds, [](Attribute attr) {
           return cast<BoolAttr>(attr).getValue();
         });
}

static bool hasLowerableAsyncDMALayout(
    FunctionOpInterface funcOp, IREE::GPU::TargetAttr target,
    VectorType vectorType,
    std::optional<int64_t> swizzleAccessElems = std::nullopt) {
  std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
      getWorkgroupSize(funcOp);
  if (!maybeWorkgroupSize) {
    return false;
  }
  std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
  if (!subgroupSize) {
    return false;
  }

  int64_t numThreads = ShapedType::getNumElements(*maybeWorkgroupSize);
  int64_t elementBits = vectorType.getElementTypeBitWidth();

  ArrayRef<int64_t> dmaSizes;
  if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
    dmaSizes = dmaSizesAttr.asArrayRef();
  }

  return succeeded(getGlobalLoadDMALayout(
      funcOp->getContext(), vectorType.getShape(), numThreads, *subgroupSize,
      elementBits, dmaSizes, swizzleAccessElems));
}

static bool
isAsyncDMAEligible(FunctionOpInterface funcOp, vector::TransferReadOp readOp,
                   IREE::GPU::TargetAttr target,
                   std::optional<int64_t> swizzleAccessElems = std::nullopt) {
  if (!target || !targetSupportsGlobalLoadDMA(target)) {
    return false;
  }
  if (!hasAllInBounds(readOp)) {
    return false;
  }

  VectorType vectorType = readOp.getVectorType();
  if (vectorType.isScalable()) {
    return false;
  }

  return hasLowerableAsyncDMALayout(funcOp, target, vectorType,
                                    swizzleAccessElems);
}

static FailureOr<IREE::Codegen::XORShuffleAttr>
deriveDMASwizzle(MLIRContext *context, IREE::GPU::TargetAttr target,
                 VectorType vectorType,
                 IREE::VectorExt::VectorLayoutInterface allocationLayout) {
  auto nestedLayout =
      dyn_cast<IREE::VectorExt::NestedLayoutAttr>(allocationLayout);
  if (!nestedLayout) {
    return failure();
  }

  int64_t elementBitWidth = vectorType.getElementTypeBitWidth();
  // The innermost element tile dimension is the contiguous access width per
  // thread during reads from the flat LDS allocation.
  int64_t accessElems = nestedLayout.getElementTile().back();
  FailureOr<XorShuffleParams> swizzleParams = getXorShuffleParamsForDMA(
      target, elementBitWidth, vectorType.getNumElements(), accessElems);
  if (failed(swizzleParams)) {
    return failure();
  }

  return IREE::Codegen::XORShuffleAttr::get(context, swizzleParams->rowElems,
                                            swizzleParams->accessElems,
                                            /*row_stride=*/0,
                                            /*per_phase=*/0);
}

/// Materialize async DMA promotion.
///
/// Assume this candidate operation:
///
/// %read = vector.transfer_read %global_memref ...
///
/// This gets promoted by transforming the IR as follows:
///
/// %alloc = bufferization.alloc_tensor <workgroup_memory>
/// %dest = iree_codegen.swizzle_hint %alloc[...]  // If valid for the layout.
/// %dma = iree_gpu.async_dma %global_memref[...] to %dest[...]
/// %barrier = iree_gpu.value_barrier %dma
/// %lds_read = vector.transfer_read %barrier
///
/// All uses of %read are replaced by %lds_read.
static FailureOr<Value> materializeAsyncDMARead(
    OpBuilder &builder, FunctionOpInterface funcOp,
    vector::TransferReadOp readOp,
    IREE::VectorExt::VectorLayoutInterface allocationLayout) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  VectorType vectorType = readOp.getVectorType();
  FailureOr<IREE::Codegen::XORShuffleAttr> swizzle = failure();
  std::optional<int64_t> swizzleAccessElems;
  if (target && targetSupportsGlobalLoadDMA(target) &&
      !vectorType.isScalable()) {
    swizzle = deriveDMASwizzle(builder.getContext(), target, vectorType,
                               allocationLayout);
    if (succeeded(swizzle)) {
      swizzleAccessElems = swizzle->getAccessElementCount();
    }
  }
  if (!isAsyncDMAEligible(funcOp, readOp, target, swizzleAccessElems)) {
    return failure();
  }
  // Synchronize before the write to shared memory to avoid stepping over
  // reads in the previous iteration of a loop. We set this barrier at the
  // start of this block.
  insertWorkgroupBarrierAtBlockStart(builder, readOp);

  Location loc = readOp.getLoc();
  builder.setInsertionPoint(readOp);
  Value dest = allocateWorkgroupTensor(builder, loc,
                                       allocationLayout.getUndistributedShape(),
                                       vectorType.getElementType());
  if (succeeded(swizzle)) {
    dest = IREE::Codegen::SwizzleHintOp::create(builder, loc, dest, *swizzle);
  }

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> zeroIndices(cast<ShapedType>(dest.getType()).getRank(),
                                 c0);

  auto dmaOp = IREE::GPU::AsyncDMAOp::create(
      builder, loc, dest.getType(), readOp.getBase(), readOp.getIndices(), dest,
      zeroIndices, TypeAttr::get(vectorType), readOp.getPermutationMapAttr(),
      readOp.getInBoundsAttr());
  auto synced =
      IREE::GPU::ValueBarrierOp::create(builder, loc, dmaOp.getResult());
  return readVectorFromTensor(builder, vectorType, synced.getResult(0));
}

/// Materialize shared memory promotion through registers.
///
/// Assume this candidate operation:
///
/// %read = vector.transfer_read %global_memref ...
///
/// This gets promoted by transforming the IR as follows:
///
/// %read = vector.transfer_read %global_memref ...
/// %global_layout = vector_ext.to_layout %read, #derived_thread_layout
/// %alloc = bufferization.alloc_tensor <workgroup_memory>
/// %lds_write = vector.transfer_write %global_layout, %alloc ...
/// %barrier = iree_gpu.value_barrier %lds_write
/// %lds_read = vector.transfer_read %barrier
///
/// All uses of %read except %global_layout are replaced by %lds_read.
static LogicalResult materializeSharedMemoryPromotion(
    OpBuilder &builder, PromotionCandidate candidate,
    IREE::VectorExt::VectorLayoutInterface allocationLayout,
    ArrayRef<int64_t> workgroupSize) {
  Value vector = candidate.vector;
  VectorType readType = cast<VectorType>(vector.getType());
  SmallVector<int64_t> elementTile = IREE::GPU::deriveThreadTileSizes(
      readType.getShape(), ShapedType::getNumElements(workgroupSize),
      readType.getElementTypeBitWidth());
  FailureOr<IREE::VectorExt::NestedLayoutAttr> readLayout =
      getDerivedThreadLayout(candidate.op->getContext(), workgroupSize,
                             readType.getShape(), elementTile);
  if (failed(readLayout)) {
    return candidate.op->emitOpError(
        "failed to derive layout for promoted read-like op");
  }
  Operation *op = candidate.op;
  // Synchronize before the write to shared memory to avoid stepping over
  // reads in the previous iteration of a loop. We set this barrier at the
  // start of this block.
  insertWorkgroupBarrierAtBlockStart(builder, op);

  builder.setInsertionPointAfter(op);

  auto toLayout = IREE::VectorExt::ToLayoutOp::create(builder, op->getLoc(),
                                                      vector, *readLayout);

  Value dest = allocateWorkgroupTensor(builder, op->getLoc(),
                                       allocationLayout.getUndistributedShape(),
                                       readType.getElementType());
  FailureOr<Value> copied =
      writeVectorToTensor(builder, op->getLoc(), toLayout.getResult(), dest);
  if (failed(copied)) {
    return failure();
  }

  auto synced =
      IREE::GPU::ValueBarrierOp::create(builder, op->getLoc(), *copied);
  Value newRead = readVectorFromTensor(builder, readType, synced.getResult(0));

  vector.replaceUsesWithIf(
      newRead, [&](OpOperand &use) { return use.getOwner() != toLayout; });
  return success();
}

/// Materialize promotion of operands to LDS.
static LogicalResult
materializeSharedMemoryPromotions(FunctionOpInterface funcOp,
                                  const PromotionTypeMap &promotionTypes,
                                  const LayoutMap &layouts) {
  SmallVector<PromotionCandidate> readsToPromote;
  funcOp.walk([&](Operation *op) {
    std::optional<PromotionCandidate> candidate =
        getPromotionCandidate(op, promotionTypes);
    if (candidate) {
      readsToPromote.push_back(*candidate);
    }
  });

  if (readsToPromote.empty()) {
    return success();
  }

  OpBuilder builder(funcOp);
  std::optional<SmallVector<int64_t>> workgroupSize = getWorkgroupSize(funcOp);
  if (!workgroupSize) {
    return funcOp.emitOpError("unable to query workgroup_size information");
  }

  for (PromotionCandidate candidate : readsToPromote) {
    Value vector = candidate.vector;
    IREE::VectorExt::VectorLayoutInterface allocationLayout =
        layouts.lookup(vector);
    if (!allocationLayout) {
      return candidate.op->emitOpError(
          "missing layout for promoted read-like op");
    }

    Attribute promotionType = candidate.promotionType;
    // If promotion via DMA was requested, first attempt to use DMA for
    // promotion. If any of the prerequisites isn't met, fall back to regular
    // global loads.
    if (isa<IREE::GPU::UseGlobalLoadDMAAttr>(promotionType)) {
      builder.setInsertionPointToStart(candidate.op->getBlock());
      gpu::BarrierOp::create(builder, candidate.op->getLoc(),
                             gpu::AddressSpace::Workgroup);

      auto readOp = cast<vector::TransferReadOp>(candidate.op);
      FailureOr<Value> dmaRead =
          materializeAsyncDMARead(builder, funcOp, readOp, allocationLayout);
      if (succeeded(dmaRead)) {
        readOp.getResult().replaceAllUsesWith(*dmaRead);
        readOp->erase();
        continue;
      }
    }

    if (failed(materializeSharedMemoryPromotion(
            builder, candidate, allocationLayout, *workgroupSize))) {
      return failure();
    }
  }
  return success();
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Remove stretching broadcasts before layout analysis — the analysis
    // asserts that broadcasts don't stretch.
    {
      RewritePatternSet patterns(funcOp.getContext());
      populateVectorLayoutCanonicalizations(patterns);
      walkAndApplyPatterns(funcOp, std::move(patterns));
    }

    // Run the promotion propagation to propagate promotion information from
    // compute ops up to the ops accessing memory.
    if (failed(validateSharedMemoryConversionAttrs(funcOp))) {
      return signalPassFailure();
    }
    PromotionTypeMap promotionTypes = analyzePromotionTypes(funcOp);

    // Remove the existing promotion information. The direct operand promotion
    // path uses the analysis result above; newly detected layout conflicts
    // below will get fresh markers.
    funcOp.walk([](IREE::VectorExt::ToLayoutOp op) {
      op.removeSharedMemoryConversionAttr();
    });

    // Run layout analysis to find additional conflict points.
    LayoutMap layouts;
    propagateVectorLayoutInfo(funcOp, layouts);

    // Mark newly-inserted to_layout ops where input/output layouts don't
    // match — these are genuine conflicts needing shared memory.
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      auto inputLayout = layouts.lookup(op.getInput());
      auto outputLayout = layouts.lookup(op.getResult());
      if (inputLayout && outputLayout &&
          inputLayout.needsSharedMemoryForConversion(outputLayout)) {
        op.setSharedMemoryConversionAttr(
            IREE::GPU::DerivedThreadConfigAttr::get(op.getContext()));
      }
    });

    // Materialize operand promotions based on the analysis.
    if (failed(materializeSharedMemoryPromotions(funcOp, promotionTypes,
                                                 layouts))) {
      return signalPassFailure();
    }

    // Materialize LDS roundtrips for layout conflicts.
    if (failed(materializeSharedMemoryConversions(funcOp))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
