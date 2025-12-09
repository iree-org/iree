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

  // Get the effective destination shape for validation.
  // When slice semantics are used, we compare source against the slice sizes.
  // Otherwise, we compare against the full init shape.
  SmallVector<int64_t> destShape;
  if (hasSliceSemantics()) {
    // Validate slice specification
    ArrayRef<int64_t> staticSizes = getStaticSizes();
    ArrayRef<int64_t> staticOffsets = getStaticOffsets();
    ArrayRef<int64_t> staticStrides = getStaticStrides();

    if (staticOffsets.size() != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " offset values, got "
             << staticOffsets.size();
    }
    if (staticSizes.size() != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " size values, got " << staticSizes.size();
    }
    if (staticStrides.size() != initShape.size()) {
      return emitOpError("expected ")
             << initShape.size() << " stride values, got "
             << staticStrides.size();
    }

    // Use sizes as the effective destination shape
    destShape.assign(staticSizes.begin(), staticSizes.end());
  } else {
    destShape.assign(initShape.begin(), initShape.end());
  }

  // Verify the contiguous (non-indexed) dimensions match between source and
  // dest (or slice sizes when using slice semantics).
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

    // Skip dynamic sizes - can't verify at compile time.
    if (ShapedType::isDynamic(size)) {
      continue;
    }

    // Check the suffix (hidden) gathering dimensions are the same in `source`
    // and `dest` (or slice sizes).
    int64_t sourceDim = sourceShape[dim];
    if (sourceDim != size) {
      if (hasSliceSemantics()) {
        return emitOpError("expected unindexed dimension ")
               << dim << " to have same length in source (" << sourceDim
               << ") and slice size (" << size << ')';
      }
      return emitOpError("expected unindexed dimension ")
             << dim << " to have same length in source (" << sourceDim
             << ") and destination (" << size << ')';
    }
  }

  return success();
}

/// Helper to print a mixed list of operands and integers (static values).
static void printDynamicIndexList(OpAsmPrinter &p, OperandRange values,
                                  ArrayRef<int64_t> staticVals) {
  unsigned dynamicIdx = 0;
  llvm::interleaveComma(staticVals, p, [&](int64_t val) {
    if (val == ShapedType::kDynamic) {
      p << values[dynamicIdx++];
    } else {
      p << val;
    }
  });
}

/// Helper to parse a mixed list of operands and integers.
static ParseResult
parseDynamicIndexList(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
                      SmallVectorImpl<int64_t> &staticVals) {
  auto parseOne = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    auto res = parser.parseOptionalOperand(operand);
    if (res.has_value() && succeeded(*res)) {
      values.push_back(operand);
      staticVals.push_back(ShapedType::kDynamic);
    } else {
      int64_t val;
      if (parser.parseInteger(val))
        return failure();
      staticVals.push_back(val);
    }
    return success();
  };
  return parser.parseCommaSeparatedList(parseOne);
}

// Custom parser for CoalescedGatherDMAOp
// Format: $source (`[` $indices^ `]`)? `into` $init
//         (`[` offset-list `]` `[` size-list `]` `[` stride-list `]`)?
//         `lane` `(` $lane `)` attr-dict `:` type(operands) (`->`
//         type($result))?
ParseResult CoalescedGatherDMAOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  OpAsmParser::UnresolvedOperand init;
  OpAsmParser::UnresolvedOperand lane;

  // Parse source
  if (parser.parseOperand(source))
    return failure();

  // Parse optional indices
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(indices) || parser.parseRSquare())
      return failure();
  }

  // Parse 'into' and init
  if (parser.parseKeyword("into") || parser.parseOperand(init))
    return failure();

  // Parse optional offset/size/stride lists
  SmallVector<OpAsmParser::UnresolvedOperand> offsets, sizes, strides;
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;

  if (succeeded(parser.parseOptionalLSquare())) {
    // We have slice semantics - parse offsets, sizes, strides
    if (parseDynamicIndexList(parser, offsets, staticOffsets) ||
        parser.parseRSquare() || parser.parseLSquare() ||
        parseDynamicIndexList(parser, sizes, staticSizes) ||
        parser.parseRSquare() || parser.parseLSquare() ||
        parseDynamicIndexList(parser, strides, staticStrides) ||
        parser.parseRSquare())
      return failure();
  }

  // Parse 'lane' '(' lane ')'
  if (parser.parseKeyword("lane") || parser.parseLParen() ||
      parser.parseOperand(lane) || parser.parseRParen())
    return failure();

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse ':' and types
  if (parser.parseColon())
    return failure();

  // Parse source type
  Type sourceType;
  if (parser.parseType(sourceType))
    return failure();

  // Parse indices types
  SmallVector<Type> indicesTypes;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (parser.parseComma())
      return failure();
    Type indexType;
    if (parser.parseType(indexType))
      return failure();
    indicesTypes.push_back(indexType);
  }

  // Parse init type
  if (parser.parseComma())
    return failure();
  Type initType;
  if (parser.parseType(initType))
    return failure();

  // Parse lane type (index)
  if (parser.parseComma())
    return failure();
  Type laneType;
  if (parser.parseType(laneType))
    return failure();

  // Parse optional result type
  Type resultType;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  // Resolve operands
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();
  if (parser.resolveOperands(indices, indicesTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  if (parser.resolveOperand(init, initType, result.operands))
    return failure();
  if (parser.resolveOperand(lane, laneType, result.operands))
    return failure();
  if (parser.resolveOperands(offsets, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();
  if (parser.resolveOperands(sizes, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();
  if (parser.resolveOperands(strides, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  // Add segment sizes
  result.addAttribute(
      CoalescedGatherDMAOp::getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(indices.size()), 1, 1,
           static_cast<int32_t>(offsets.size()),
           static_cast<int32_t>(sizes.size()),
           static_cast<int32_t>(strides.size())}));

  // Add static offset/size/stride attributes
  result.addAttribute(
      CoalescedGatherDMAOp::getStaticOffsetsAttrName(result.name),
      parser.getBuilder().getDenseI64ArrayAttr(staticOffsets));
  result.addAttribute(CoalescedGatherDMAOp::getStaticSizesAttrName(result.name),
                      parser.getBuilder().getDenseI64ArrayAttr(staticSizes));
  result.addAttribute(
      CoalescedGatherDMAOp::getStaticStridesAttrName(result.name),
      parser.getBuilder().getDenseI64ArrayAttr(staticStrides));

  return success();
}

// Custom printer for CoalescedGatherDMAOp
void CoalescedGatherDMAOp::print(OpAsmPrinter &p) {
  p << ' ' << getSource();

  // Print optional indices
  if (!getIndices().empty()) {
    p << '[';
    llvm::interleaveComma(getIndices(), p);
    p << ']';
  }

  p << " into " << getInit();

  // Print optional slice offsets/sizes/strides
  if (hasSliceSemantics()) {
    p << " [";
    printDynamicIndexList(p, getOffsets(), getStaticOffsets());
    p << "] [";
    printDynamicIndexList(p, getSizes(), getStaticSizes());
    p << "] [";
    printDynamicIndexList(p, getStrides(), getStaticStrides());
    p << ']';
  }

  p << " lane(" << getLane() << ')';

  // Print attributes (excluding operand_segment_sizes and static_* attrs)
  SmallVector<StringRef> elidedAttrs = {
      getOperandSegmentSizesAttrName(), getStaticOffsetsAttrName(),
      getStaticSizesAttrName(), getStaticStridesAttrName()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

  // Print types
  p << " : " << getSource().getType();
  for (Value index : getIndices()) {
    p << ", " << index.getType();
  }
  p << ", " << getInit().getType();
  p << ", " << getLane().getType();

  // Print optional result type
  if (getResult()) {
    p << " -> " << getResult().getType();
  }
}

// Builder with OpFoldResult for offsets/sizes/strides
void CoalescedGatherDMAOp::build(OpBuilder &builder, OperationState &result,
                                 Type resultType, Value source,
                                 ValueRange indices, Value init, Value lane,
                                 ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes,
                                 ArrayRef<OpFoldResult> strides,
                                 ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  result.addOperands(source);
  result.addOperands(indices);
  result.addOperands(init);
  result.addOperands(lane);
  result.addOperands(dynamicOffsets);
  result.addOperands(dynamicSizes);
  result.addOperands(dynamicStrides);
  result.addAttribute(
      CoalescedGatherDMAOp::getOperandSegmentSizesAttrName(result.name),
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(indices.size()), 1, 1,
           static_cast<int32_t>(dynamicOffsets.size()),
           static_cast<int32_t>(dynamicSizes.size()),
           static_cast<int32_t>(dynamicStrides.size())}));
  result.addAttribute(
      CoalescedGatherDMAOp::getStaticOffsetsAttrName(result.name),
      builder.getDenseI64ArrayAttr(staticOffsets));
  result.addAttribute(CoalescedGatherDMAOp::getStaticSizesAttrName(result.name),
                      builder.getDenseI64ArrayAttr(staticSizes));
  result.addAttribute(
      CoalescedGatherDMAOp::getStaticStridesAttrName(result.name),
      builder.getDenseI64ArrayAttr(staticStrides));
  if (resultType)
    result.addTypes(resultType);
  result.addAttributes(attrs);
}

} // namespace mlir::iree_compiler::IREE::GPU
