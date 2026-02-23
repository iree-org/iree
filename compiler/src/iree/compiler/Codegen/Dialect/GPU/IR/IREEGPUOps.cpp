// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::GPU {
//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

// Build a BarrierRegionOp with an empty.
void BarrierRegionOp::build(OpBuilder &b, OperationState &result,
                            TypeRange resultTypes, ValueRange inputs) {
  result.addOperands(inputs);
  (void)result.addRegion();
  result.addTypes(resultTypes);
  SmallVector<Location> blockArgLocs(inputs.size(), result.location);

  Region *region = result.regions[0].get();

  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), inputs.getTypes(), blockArgLocs);
}

LogicalResult BarrierRegionOp::verify() { return success(); }

LogicalResult BarrierRegionOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != getNumOperands()) {
    return emitError(
        "expected the block argument count to match operand count");
  }

  if (!llvm::all_of_zip(block.getArgumentTypes(), getOperandTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected block argument types to match operand types");
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != getNumResults()) {
    return emitOpError(
        "expected body to yield same number of values as results");
  }

  if (!llvm::all_of_zip(yieldOp->getOperandTypes(), getResultTypes(),
                        [](Type a, Type b) { return a == b; })) {
    return emitError("expected yielded value types to match result types");
  }

  return success();
}

SmallVector<Region *> BarrierRegionOp::getHoistableRegions() {
  return {&getBodyRegion()};
}

// We only need it to be memory effect free, not speculatable, as this
// region is guaranteed to execute.
bool BarrierRegionOp::isHoistable(Operation *op) {
  return isMemoryEffectFree(op);
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

void ValueBarrierOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange input) {
  result.addOperands(input);
  result.addTypes(llvm::map_range(input, [](Value v) { return v.getType(); }));
}

LogicalResult ValueBarrierOp::verify() {
  if (getNumOperands() == 0) {
    return emitOpError("Atleast one input required");
  }

  // Make sure we either have all tensors or all vectors.
  if (hasTensorSemantics()) {
    bool allTensor =
        llvm::all_of(getInputTypes(), llvm::IsaPred<RankedTensorType>);
    if (!allTensor) {
      return emitOpError(
          "All inputs should be either of tensor or vector type");
    }
    return success();
  }

  bool allVector = llvm::all_of(getInputTypes(), llvm::IsaPred<VectorType>);
  if (!allVector) {
    return emitOpError("All inputs should be either of tensor or vector type");
  }

  return success();
}

// AMD Specific Operations

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

static RankedTensorType getMaximumStaticType(tensor::CastOp castOp) {
  auto inputType = dyn_cast<RankedTensorType>(castOp.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
  if (!inputType || !resultType) {
    return RankedTensorType();
  }

  assert(inputType.getRank() == resultType.getRank() &&
         "Rank must match for ranked -> ranked cast");

  SmallVector<int64_t> join;
  join.reserve(inputType.getRank());
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      join.push_back(resultType.getDimSize(i));
      continue;
    }
    if (resultType.isDynamicDim(i)) {
      join.push_back(inputType.getDimSize(i));
      continue;
    }

    // Cast verifier requires that static sizes match.
    join.push_back(inputType.getDimSize(i));
  }
  return RankedTensorType::get(join, inputType.getElementType());
}

struct FoldBufferCastOfTensorCast final
    : OpRewritePattern<BufferResourceCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(BufferResourceCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the cast increases the amount of available static info.
    auto tensorCast = castOp.getInput().getDefiningOp<tensor::CastOp>();
    if (!tensorCast) {
      return failure();
    }

    RankedTensorType maxStaticType = getMaximumStaticType(tensorCast);
    if (!maxStaticType || maxStaticType == castOp.getInput().getType()) {
      return failure();
    }

    Value newSource = tensorCast.getSource();
    if (newSource.getType() != maxStaticType) {
      // Cast to the type with maximum static information if the input and
      // result types contain different static info.
      newSource = tensor::CastOp::create(rewriter, castOp.getLoc(),
                                         maxStaticType, newSource);
    }
    auto newBufferCast = IREE::GPU::BufferResourceCastOp::create(
        rewriter, castOp.getLoc(), maxStaticType, newSource,
        castOp.getCacheSwizzleStride());
    newBufferCast->setDiscardableAttrs(castOp->getDiscardableAttrDictionary());

    // Cast back to the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        castOp, castOp.getResult().getType(), newBufferCast);
    return success();
  };
};

void BufferResourceCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *ctx) {
  results.add<FoldBufferCastOfTensorCast>(ctx);
}

//===----------------------------------------------------------------------===//
// CoalescedGatherDMAOp
//===----------------------------------------------------------------------===//

// ParallelCombiningOpInterface implementation
MutableOperandRange CoalescedGatherDMAOp::getUpdatedDestinations() {
  // Only relevant for tensor operands
  if (!isa<RankedTensorType>(getInit().getType())) {
    return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
  }
  // Return the init operand as the destination being updated
  return getInitMutable();
}

Operation *CoalescedGatherDMAOp::getIteratingParent() {
  // Only relevant for tensor operands
  if (!isa<RankedTensorType>(getInit().getType())) {
    return nullptr;
  }
  // Return the parent scf.forall operation
  return getOperation()->getParentOfType<scf::ForallOp>();
}

void CoalescedGatherDMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Get the OpOperand pointers for source and init
  // Operand layout: source, indices (variadic), init, lane
  unsigned numOperands = getOperation()->getNumOperands();
  unsigned laneOperandIdx = numOperands - 1;
  unsigned initOperandIdx = laneOperandIdx - 1;
  unsigned sourceOperandIdx = 0;

  Value source = getSource();
  Value init = getInit();

  // The operation reads from the source.
  if (isa<MemRefType>(source.getType())) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(sourceOperandIdx),
                         SideEffects::DefaultResource::get());
  }

  // For memref form, the operation writes to init (side effect)
  // For tensor form with result, the write is captured in the result value
  // For tensor form without result (combiner case in forall.in_parallel),
  // we must declare a write effect to prevent DCE from eliminating the op.
  if (isa<MemRefType>(init.getType())) {
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(initOperandIdx),
                         SideEffects::DefaultResource::get());
  } else if (isa<RankedTensorType>(init.getType()) &&
             getOperation()->getNumResults() == 0) {
    // Tensor combiner case: declare write effect to prevent DCE.
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(initOperandIdx),
                         SideEffects::DefaultResource::get());
  }
}

LogicalResult CoalescedGatherDMAOp::verify() {
  TypedValue<ShapedType> init = getInit();
  auto initType = init.getType();

  bool hasTensor = isa<RankedTensorType>(initType);
  bool hasMemRef = isa<MemRefType>(initType);

  if (!hasTensor && !hasMemRef) {
    return emitOpError("init type must either be a tensor or a memref");
  }

  auto initShapedType = cast<ShapedType>(initType);
  auto sourceType = cast<ShapedType>(getSource().getType());
  ArrayRef<int64_t> initShape = initShapedType.getShape();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();

  if (hasTensor && !isa<RankedTensorType>(sourceType)) {
    return emitOpError("source must be tensor when init is tensor");
  }
  if (hasMemRef && !isa<MemRefType>(sourceType)) {
    return emitOpError("source must be memref when init is memref");
  }

  OperandRange indices = getIndices();

  if (indices.size() > initShape.size()) {
    return emitOpError("number of indices (")
           << indices.size() << ") cannot exceed destination rank ("
           << initShape.size() << ")";
  }

  if (indices.size() > sourceShape.size()) {
    return emitOpError("number of indices (")
           << indices.size() << ") cannot exceed source rank ("
           << sourceShape.size() << ")";
  }

  // Make sure indices have no dynamic shapes.
  for (auto [i, indexVal] : llvm::enumerate(indices)) {
    auto indexType = cast<ShapedType>(indexVal.getType());
    for (auto dim : indexType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        return emitOpError("expected index ") << i << " to have static shape";
      }
    }
  }

  // For gather operations with indices, all index vectors should have the same
  // length equal to the batch size (first dimension of destination). This is
  // validated here so that lowering passes can rely on these constraints
  // without duplicating the checks.
  if (!indices.empty()) {
    // Verify all index vectors are 1D and have the same length.
    auto firstIndexShape = cast<ShapedType>(indices[0].getType()).getShape();
    if (firstIndexShape.size() != 1) {
      return emitOpError("expected index 0 to be a 1-D tensor or vector");
    }
    int64_t batchSize = firstIndexShape.front();

    for (auto [i, indexVal] : llvm::enumerate(indices)) {
      auto indexShape = cast<ShapedType>(indexVal.getType()).getShape();
      if (indexShape.size() != 1) {
        return emitOpError("expected index ")
               << i << " to be a 1-D tensor or vector";
      }
      if (indexShape.front() != batchSize) {
        return emitOpError(
                   "expected all index vectors to have the same length; ")
               << "index " << i << " has length " << indexShape.front()
               << " but expected " << batchSize;
      }
    }

    // The batch size should match the first dimension of the destination.
    if (!initShape.empty() && batchSize != initShape[0]) {
      return emitOpError("expected batch size (length of index vectors: ")
             << batchSize << ") to match first destination dimension ("
             << initShape[0] << ")";
    }
  }

  // Verify the contiguous (non-indexed) dimensions match between source and
  // dest, unless in_bounds allows OOB reads for that dimension.
  std::optional<ArrayAttr> inBoundsAttr = getInBounds();
  for (auto [dim, size] : llvm::enumerate(initShape)) {
    if (dim >= sourceShape.size()) {
      return emitOpError("expected source to have at least ")
             << (dim + 1) << " dimensions when destination has rank "
             << initShape.size();
    }

    // Skip indexed dimensions - they're validated above.
    if (dim < indices.size()) {
      continue;
    }

    // If in_bounds is present and this dimension allows OOB (in_bounds=false),
    // skip the size matching check. The source may be smaller than init along
    // this dimension, and reads beyond the source extent return zero.
    if (inBoundsAttr) {
      auto inBoundsArray = *inBoundsAttr;
      if (dim < inBoundsArray.size()) {
        bool dimInBounds = cast<BoolAttr>(inBoundsArray[dim]).getValue();
        if (!dimInBounds) {
          continue; // OOB allowed, skip size check
        }
      }
    }

    // Check the suffix (hidden) gathering dimensions are the same in `source`
    // and `init`.
    int64_t sourceDim = sourceShape[dim];
    if (sourceDim != size) {
      return emitOpError("expected unindexed dimension ")
             << dim << " to have same length in source (" << sourceDim
             << ") and destination (" << size << ')';
    }
  }

  // Validate in_bounds attribute if present.
  if (std::optional<ArrayAttr> inBoundsAttr = getInBounds()) {
    int64_t initRank = initShapedType.getRank();
    if (static_cast<int64_t>(inBoundsAttr->size()) != initRank) {
      return emitOpError("in_bounds array size (")
             << inBoundsAttr->size() << ") must match init rank (" << initRank
             << ")";
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
