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
#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
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

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor
// and then writes the vector into the allocated tensor. This allocation is
// always static as vectors are currently always static where this is used.
static FailureOr<Value> allocateTensorAndWriteVector(OpBuilder &b, Location loc,
                                                     Value vector,
                                                     ArrayRef<int64_t> shape) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType = RankedTensorType::get(
      shape, vectorType.getElementType(), sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocTensorOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  llvm::Repeated<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = vector::TransferWriteOp::create(b, loc, vector, allocTensorOp,
                                                 indices, inBounds)
                     .getResult();
  return copied;
}

static FailureOr<Value> allocateTensorAndWriteVector(OpBuilder &b, Location loc,
                                                     Value vector) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  return allocateTensorAndWriteVector(b, loc, vector, vectorType.getShape());
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
    builder.setInsertionPointToStart(op->getBlock());
    gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);

    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();
    // TODO: Since we know the read/write layout for this memory, we can get
    // optimal swizzling here. Figure out how to do that.
    FailureOr<Value> ret =
        allocateTensorAndWriteVector(builder, op->getLoc(), operand.get());
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
  Operation *op;
  Value vector;
};

/// Find promotion candidates, i.e., operations that load data from memory and
/// were reached by a promotion type during analysis propagation.
static std::optional<PromotionCandidate>
getPromotionCandidate(Operation *op, const PromotionTypeMap &promotionTypes) {
  auto shouldPromoteCandidate =
      [&](Operation *op, Value vector) -> std::optional<PromotionCandidate> {
    auto promotionType = promotionTypes.find(vector);
    if (promotionType == promotionTypes.end()) {
      return std::nullopt;
    }

    // TODO: Currently only handles derived_thread_config promotion types,
    // extend to other promotion types.
    if (!isa<IREE::GPU::DerivedThreadConfigAttr>(promotionType->second)) {
      return std::nullopt;
    }
    return PromotionCandidate{op, vector};
  };
  return TypeSwitch<Operation *, std::optional<PromotionCandidate>>(op)
      .Case([&](vector::TransferReadOp read) {
        return shouldPromoteCandidate(read.getOperation(), read.getVector());
      })
      .Case([&](vector::GatherOp gather) {
        return shouldPromoteCandidate(gather.getOperation(),
                                      gather.getResult());
      })
      .Default(std::nullopt);
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
  // Assume this candidate operation:
  //
  // %read = vector.transfer_read %global_memref ...
  //
  // This gets promoted by transforming the IR as follows:
  //
  // %read = vector.transfer_read %global_memref ...
  // %global_layout = vector_ext.to_layout %read, #derived_thread_layout
  // %alloc = bufferization.alloc_tensor <workgroup_memory>
  // %lds_write = vector.transfer_write %global_layout, %alloc ...
  // %barrier = iree_gpu.value_barrier %lds_write
  // %lds_read = transfer_read %barrier
  //
  // All uses of %read except %global_layout will be replaced by %lds_read.
  for (PromotionCandidate candidate : readsToPromote) {
    Value vector = candidate.vector;
    IREE::VectorExt::VectorLayoutInterface allocationLayout =
        layouts.lookup(vector);
    if (!allocationLayout) {
      return candidate.op->emitOpError(
          "missing layout for promoted read-like op");
    }

    VectorType readType = cast<VectorType>(vector.getType());
    SmallVector<int64_t> elementTile = IREE::GPU::deriveThreadTileSizes(
        readType.getShape(), ShapedType::getNumElements(*workgroupSize),
        readType.getElementTypeBitWidth());
    FailureOr<IREE::VectorExt::NestedLayoutAttr> readLayout =
        getDerivedThreadLayout(candidate.op->getContext(), *workgroupSize,
                               readType.getShape(), elementTile);
    if (failed(readLayout)) {
      return candidate.op->emitOpError(
          "failed to derive layout for promoted read-like op");
    }
    Operation *op = candidate.op;
    builder.setInsertionPointAfter(op);

    auto toLayout = IREE::VectorExt::ToLayoutOp::create(builder, op->getLoc(),
                                                        vector, *readLayout);

    FailureOr<Value> copied = allocateTensorAndWriteVector(
        builder, op->getLoc(), toLayout.getResult(),
        allocationLayout.getUndistributedShape());
    if (failed(copied)) {
      return failure();
    }

    auto synced =
        IREE::GPU::ValueBarrierOp::create(builder, op->getLoc(), *copied);
    Value newRead =
        readVectorFromTensor(builder, readType, synced.getResult(0));

    vector.replaceUsesWithIf(
        newRead, [&](OpOperand &use) { return use.getOwner() != toLayout; });
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
    PromotionTypeMap promotionTypes = analyzePromotionTypes(funcOp);

    // Remove the existing promotion information. All those operations will have
    // their promotion materialized based on the analysis above.
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
