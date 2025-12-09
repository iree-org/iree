// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
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

    // The batch size should match the first dimension of the destination
    // (or slice size if sliced).
    ArrayRef<int64_t> staticSizesForBatch = getStaticSizes();
    int64_t destBatchDim = staticSizesForBatch.empty()
                               ? (initShape.empty() ? 0 : initShape[0])
                               : staticSizesForBatch[0];
    if (destBatchDim != 0 && batchSize != destBatchDim) {
      return emitOpError("expected batch size (length of index vectors: ")
             << batchSize << ") to match first destination dimension ("
             << destBatchDim << ")";
    }
  }

  // Verify offsets/sizes/strides if present.
  ArrayRef<int64_t> staticOffsets = getStaticOffsets();
  ArrayRef<int64_t> staticSizes = getStaticSizes();
  ArrayRef<int64_t> staticStrides = getStaticStrides();
  bool hasSliceParams = !staticOffsets.empty();
  if (hasSliceParams) {
    unsigned initRank = initShapedType.getRank();
    if (staticOffsets.size() != initRank || staticSizes.size() != initRank ||
        staticStrides.size() != initRank) {
      return emitOpError("expected offsets, sizes, and strides to have ")
             << initRank << " elements to match init rank, but got "
             << staticOffsets.size() << ", " << staticSizes.size() << ", "
             << staticStrides.size();
    }
  }

  // For dimension matching, use slice sizes if present, otherwise use init
  // shape.
  SmallVector<int64_t> destShape;
  if (hasSliceParams) {
    destShape = llvm::to_vector(staticSizes);
  } else {
    destShape = llvm::to_vector(initShape);
  }

  // Verify the contiguous (non-indexed) dimensions match between source and
  // dest (or slice sizes if sliced).
  for (auto [dim, size] : llvm::enumerate(destShape)) {
    if (dim >= sourceShape.size()) {
      return emitOpError("expected source to have at least ")
             << (dim + 1) << " dimensions when destination has rank "
             << destShape.size();
    }

    // Skip indexed dimensions - they're validated above.
    if (dim < indices.size()) {
      continue;
    }

    // Skip dynamic sizes in slice parameters.
    if (ShapedType::isDynamic(size)) {
      continue;
    }

    // Check the suffix (hidden) gathering dimensions are the same in `source`
    // and dest (or slice sizes).
    int64_t sourceDim = sourceShape[dim];
    if (sourceDim != size) {
      return emitOpError("expected unindexed dimension ")
             << dim << " to have same length in source (" << sourceDim
             << ") and destination (" << size << ')';
    }
  }

  return success();
}

// Builder without slice parameters (backward compatible).
void CoalescedGatherDMAOp::build(OpBuilder &b, OperationState &result,
                                 Type resultType, Value source,
                                 ValueRange indices, Value init, Value lane) {
  build(b, result, resultType, source, indices, init, lane,
        /*offsets=*/ArrayRef<OpFoldResult>{},
        /*sizes=*/ArrayRef<OpFoldResult>{},
        /*strides=*/ArrayRef<OpFoldResult>{});
}

// Builder with mixed static and dynamic slice entries.
void CoalescedGatherDMAOp::build(OpBuilder &b, OperationState &result,
                                 Type resultType, Value source,
                                 ValueRange indices, Value init, Value lane,
                                 ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes,
                                 ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  build(b, result, resultType, source, indices, init, lane, dynamicOffsets,
        dynamicSizes, dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

SmallVector<OpFoldResult> CoalescedGatherDMAOp::getMixedOffsets() {
  Builder b(getContext());
  return getMixedValues(getStaticOffsets(), getOffsets(), b);
}

SmallVector<OpFoldResult> CoalescedGatherDMAOp::getMixedSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticSizes(), getSizes(), b);
}

SmallVector<OpFoldResult> CoalescedGatherDMAOp::getMixedStrides() {
  Builder b(getContext());
  return getMixedValues(getStaticStrides(), getStrides(), b);
}

// Custom parser for CoalescedGatherDMAOp.
// Format: source ('[' indices ']')? 'into' init
//         ('[' offsets ']' '[' sizes ']' '[' strides ']')?
//         'lane' '(' lane ')' ':' type (',' type)? 'into' type ('->' type)?
ParseResult CoalescedGatherDMAOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  OpAsmParser::UnresolvedOperand init;
  OpAsmParser::UnresolvedOperand lane;
  SmallVector<OpAsmParser::UnresolvedOperand> offsets, sizes, strides;
  DenseI64ArrayAttr staticOffsets, staticSizes, staticStrides;
  Type sourceType, initType, resultType;
  SmallVector<Type> indicesTypes;

  // Parse: source
  if (parser.parseOperand(source))
    return failure();

  // Parse optional: '[' indices ']'
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(indices) || parser.parseRSquare())
      return failure();
  }

  // Parse: 'into' init
  if (parser.parseKeyword("into") || parser.parseOperand(init))
    return failure();

  // Parse optional: '[' offsets ']' '[' sizes ']' '[' strides ']'
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parseDynamicIndexList(parser, offsets, staticOffsets) ||
        parser.parseRSquare())
      return failure();
    if (parser.parseLSquare() ||
        parseDynamicIndexList(parser, sizes, staticSizes) ||
        parser.parseRSquare())
      return failure();
    if (parser.parseLSquare() ||
        parseDynamicIndexList(parser, strides, staticStrides) ||
        parser.parseRSquare())
      return failure();
  } else {
    // No slice parameters - use empty arrays.
    staticOffsets = parser.getBuilder().getDenseI64ArrayAttr({});
    staticSizes = parser.getBuilder().getDenseI64ArrayAttr({});
    staticStrides = parser.getBuilder().getDenseI64ArrayAttr({});
  }

  // Parse: 'lane' '(' lane ')'
  if (parser.parseKeyword("lane") || parser.parseLParen() ||
      parser.parseOperand(lane) || parser.parseRParen())
    return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse: ':' sourceType
  if (parser.parseColon() || parser.parseType(sourceType))
    return failure();

  // Parse optional: ',' indicesTypes
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseTypeList(indicesTypes))
      return failure();
  }

  // Parse: 'into' initType
  if (parser.parseKeyword("into") || parser.parseType(initType))
    return failure();

  // Parse optional: '->' resultType
  bool hasResult = succeeded(parser.parseOptionalArrow());
  if (hasResult) {
    if (parser.parseType(resultType))
      return failure();
  }

  // Resolve operands.
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();
  for (auto [idx, indicesType] : llvm::zip(indices, indicesTypes)) {
    if (parser.resolveOperand(idx, indicesType, result.operands))
      return failure();
  }
  if (parser.resolveOperand(init, initType, result.operands))
    return failure();
  if (parser.resolveOperand(lane, parser.getBuilder().getIndexType(),
                            result.operands))
    return failure();
  for (auto &offset : offsets) {
    if (parser.resolveOperand(offset, parser.getBuilder().getIndexType(),
                              result.operands))
      return failure();
  }
  for (auto &size : sizes) {
    if (parser.resolveOperand(size, parser.getBuilder().getIndexType(),
                              result.operands))
      return failure();
  }
  for (auto &stride : strides) {
    if (parser.resolveOperand(stride, parser.getBuilder().getIndexType(),
                              result.operands))
      return failure();
  }

  // Add attributes.
  result.addAttribute(getStaticOffsetsAttrName(result.name), staticOffsets);
  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  result.addAttribute(getStaticStridesAttrName(result.name), staticStrides);

  // Add operand segment sizes.
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(indices.size()), 1, 1,
                           static_cast<int32_t>(offsets.size()),
                           static_cast<int32_t>(sizes.size()),
                           static_cast<int32_t>(strides.size())}));

  // Add result type if present.
  if (hasResult) {
    result.addTypes(resultType);
  }

  return success();
}

// Custom printer for CoalescedGatherDMAOp.
void CoalescedGatherDMAOp::print(OpAsmPrinter &p) {
  p << " " << getSource();
  if (!getIndices().empty()) {
    p << "[" << getIndices() << "]";
  }
  p << " into " << getInit();

  // Print slice parameters if present.
  if (hasSlice()) {
    p << "[";
    printDynamicIndexList(p, *this, getOffsets(), getStaticOffsets());
    p << "] [";
    printDynamicIndexList(p, *this, getSizes(), getStaticSizes());
    p << "] [";
    printDynamicIndexList(p, *this, getStrides(), getStaticStrides());
    p << "]";
  }

  p << " lane(" << getLane() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getOperandSegmentSizeAttr(),
                           getStaticOffsetsAttrName(), getStaticSizesAttrName(),
                           getStaticStridesAttrName()});
  p << " : " << getSource().getType();
  if (!getIndices().empty()) {
    p << ", ";
    llvm::interleaveComma(getIndices().getTypes(), p);
  }
  p << " into " << getInit().getType();
  if (getResult()) {
    p << " -> " << getResult().getType();
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
