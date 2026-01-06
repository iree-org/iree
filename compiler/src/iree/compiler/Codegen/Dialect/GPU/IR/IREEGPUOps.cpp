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
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::GPU;

//===----------------------------------------------------------------------===//
// Custom directive for optional slice (offsets/sizes/strides)
// Prints nothing when empty, or [offsets] [sizes] [strides] when present
// All values are dynamic (SSA values).
//===----------------------------------------------------------------------===//

static ParseResult
parseOptionalSlice(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sizes,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &strides) {
  // Try to parse [offsets] [sizes] [strides] - if first '[' fails, it's empty
  if (failed(parser.parseOptionalLSquare()))
    return success(); // Empty - no slice semantics

  // Parse offsets
  if (parser.parseOperandList(offsets) || parser.parseRSquare())
    return failure();

  // Parse sizes [...]
  if (parser.parseLSquare() || parser.parseOperandList(sizes) ||
      parser.parseRSquare())
    return failure();

  // Parse strides [...]
  if (parser.parseLSquare() || parser.parseOperandList(strides) ||
      parser.parseRSquare())
    return failure();

  return success();
}

static void printOptionalSlice(OpAsmPrinter &p, Operation *op,
                               OperandRange offsets, OperandRange sizes,
                               OperandRange strides) {
  if (offsets.empty())
    return; // No slice - print nothing

  p << "[";
  llvm::interleaveComma(offsets, p);
  p << "] [";
  llvm::interleaveComma(sizes, p);
  p << "] [";
  llvm::interleaveComma(strides, p);
  p << "]";
}

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

  // Validate slice semantics if present.
  if (hasSliceSemantics()) {
    size_t numOffsets = getOffsets().size();
    size_t numSizes = getSizes().size();
    size_t numStrides = getStrides().size();

    if (numOffsets != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " offset values, got " << numOffsets;
    }
    if (numSizes != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " size values, got " << numSizes;
    }
    if (numStrides != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " stride values, got " << numStrides;
    }
    // With all-dynamic slice values, we can't verify shapes at compile time.
    // The shape verification will happen at runtime.
  } else {
    // Without slice semantics, verify source and init shapes match
    // for non-indexed dimensions.
    for (size_t dim = indices.size(); dim < initShape.size(); ++dim) {
      if (dim >= sourceShape.size()) {
        return emitOpError("expected source to have at least ")
               << (dim + 1) << " dimensions when destination has rank "
               << initShape.size();
      }

      int64_t sourceDim = sourceShape[dim];
      int64_t destDim = initShape[dim];

      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(sourceDim) || ShapedType::isDynamic(destDim)) {
        continue;
      }

      if (sourceDim != destDim) {
        return emitOpError("expected unindexed dimension ")
               << dim << " to have same length in source (" << sourceDim
               << ") and destination (" << destDim << ')';
      }
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::GPU
