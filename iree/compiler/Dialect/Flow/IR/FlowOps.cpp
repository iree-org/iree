// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Op utilities used within the Flow dialect
//===----------------------------------------------------------------------===//

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// of the dynamic dimensions in |values|.
static LogicalResult verifyOpDynamicDims(Operation *op, ValueRange values,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
      requiredCount += shapedType.getNumDynamicDims();
    } else if (auto tensorType =
                   value.getType().dyn_cast<DispatchTensorType>()) {
      requiredCount += tensorType.getNumDynamicDims();
    }
  }
  if (dynamicDims.size() != requiredCount) {
    return op->emitOpError()
           << "value set has " << requiredCount
           << " dynamic dimensions but only " << dynamicDims.size()
           << " dimension values are attached";
  }
  return success();
}

// Gets the dropped dimensions for `flow.dispatch.tensor.load/store`.
static llvm::SmallBitVector getDroppedDimsImpl(
    RankedTensorType slicedObjectType, ArrayRef<OpFoldResult> mixedSizes) {
  ArrayRef<int64_t> resultShape = slicedObjectType.getShape();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  unsigned shapePos = 0;
  for (const auto &size : enumerate(mixedSizes)) {
    Optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || sizeVal.getValue() != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
  }
  return droppedDims;
}

/// Helper function to create `AffineExpr` from `OpFoldResult`. If the
/// `OpFoldResult` is a `Value`, creates a `AffineSymbolExpr` and appends it to
/// `symbols`.
static AffineExpr getAffineExpr(OpFoldResult ofr, SmallVector<Value> &symbols) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    return getAffineConstantExpr(attr.cast<IntegerAttr>().getInt(),
                                 attr.getContext());
  }
  Value v = ofr.get<Value>();
  AffineExpr expr = getAffineSymbolExpr(symbols.size(), v.getContext());
  symbols.push_back(v);
  return expr;
}
/// Converts an `AffineExpr` to `OpFoldResult` by generating an `affine.apply`
/// operation.
static OpFoldResult getOpFoldResult(OpBuilder &builder, Location loc,
                                    AffineExpr expr,
                                    SmallVector<Value> &symbols) {
  AffineMap m = AffineMap::get(0, symbols.size(), expr);
  return applyMapToValues(builder, loc, m, symbols)[0];
}

/// Methods to build the Affine Expr for arithmetic operations.
static AffineExpr add(AffineExpr expr, OpFoldResult ofr,
                      SmallVector<Value> &symbols) {
  return expr + getAffineExpr(ofr, symbols);
}
static AffineExpr add(OpFoldResult lhs, OpFoldResult rhs,
                      SmallVector<Value> &symbols) {
  return getAffineExpr(lhs, symbols) + getAffineExpr(rhs, symbols);
}
static AffineExpr mul(AffineExpr expr, OpFoldResult ofr,
                      SmallVector<Value> &symbols) {
  return expr * getAffineExpr(ofr, symbols);
}
static AffineExpr mul(OpFoldResult lhs, OpFoldResult rhs,
                      SmallVector<Value> &symbols) {
  return getAffineExpr(lhs, symbols) * getAffineExpr(rhs, symbols);
}

/// Returns the `hal.interface.binding` a value comes from.
static Optional<BlockArgument> getBindingArgument(Value v) {
  if (BlockArgument blockArg = v.dyn_cast<BlockArgument>()) {
    if (isa<IREE::Flow::DispatchWorkgroupsOp>(
            blockArg.getOwner()->getParentOp())) {
      return blockArg;
    }
    return llvm::None;
  }
  Operation *definingOp = v.getDefiningOp();
  if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(definingOp)) {
    return getBindingArgument(loadOp.source());
  }
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tie_shape
//===----------------------------------------------------------------------===//

LogicalResult DispatchTieShapeOp::verify() {
  if (failed(
          verifyOpDynamicDims(getOperation(), {operand()}, dynamic_dims()))) {
    return failure();
  }
  return success();
}

LogicalResult DispatchTieShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<Value> shape;
  unsigned dynamicIdx = 0;
  auto tensorType = result().getType().cast<IREE::Flow::DispatchTensorType>();
  for (int64_t dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamicSize) {
      shape.push_back(dynamic_dims()[dynamicIdx++]);
    } else {
      shape.push_back(b.create<arith::ConstantIndexOp>(getLoc(), dim));
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorLoadOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {source()}, source_dims()))) {
    return failure();
  }
  return success();
}

/// Extracts static and dynamic values from list of `OpFoldResult`.
static void processMixedOperands(ArrayRef<OpFoldResult> valueOrAttrs,
                                 SmallVectorImpl<Value> &dynamicValues,
                                 SmallVectorImpl<int64_t> &staticValues,
                                 int64_t dynamicIndexValue) {
  for (OpFoldResult valueOrAttr : valueOrAttrs) {
    if (auto value = valueOrAttr.dyn_cast<Value>()) {
      dynamicValues.push_back(value);
      staticValues.push_back(dynamicIndexValue);
    } else {
      auto operandValue =
          valueOrAttr.dyn_cast<Attribute>().cast<IntegerAttr>().getInt();
      staticValues.push_back(operandValue);
    }
  }
}

/// Implements default offset, sizes and strides, for
/// `flow.dispatch.tensor.load/store` ops. When no offsets, sizes and strides
/// are specified, the offsets are all zeros, sizes are same as the dispatch
/// tensor and strides are all 1.
static void getDefaultOffsetSizeAndStrides(
    OpBuilder &builder, IREE::Flow::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) {
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);
  int64_t dispatchTensorRank = dispatchTensorType.getRank();
  offsets.assign(dispatchTensorRank, zeroAttr);
  strides.assign(dispatchTensorRank, oneAttr);
  sizes.resize(dispatchTensorRank);
  unsigned pos = 0;
  for (auto dim : llvm::enumerate(dispatchTensorType.getShape())) {
    if (ShapedType::isDynamic(dim.value())) {
      assert(pos < dynamicDims.size() && "missing dynamic dims specifications");
      sizes[dim.index()] = dynamicDims[pos++];
      continue;
    }
    sizes[dim.index()] = builder.getI64IntegerAttr(dim.value());
  }
  return;
}

RankedTensorType DispatchTensorLoadOp::inferRankReducedResultType(
    unsigned resultRank, IREE::Flow::DispatchTensorType sourceType,
    ArrayRef<OpFoldResult> mixedSizes) {
  // This is using logic from
  // `tensor::ExtractSliceOp::inferRankReducedResultType`. Eventually just use
  // that.
  auto shape = llvm::to_vector<4>(
      llvm::map_range(mixedSizes, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return attr.cast<IntegerAttr>().getInt();
        }
        return DispatchTensorType::kDynamicSize;
      }));
  auto inferredType = RankedTensorType::get(shape, sourceType.getElementType());
  int rankDiff = sourceType.getRank() - resultRank;
  if (rankDiff > 0) {
    llvm::SmallBitVector dimsToProject =
        mlir::getPositionsOfShapeOne(rankDiff, shape);
    SmallVector<int64_t> projectedShape;
    for (unsigned pos = 0, e = shape.size(); pos < e; ++pos) {
      if (!dimsToProject.test(pos)) {
        projectedShape.push_back(shape[pos]);
      }
    }
    inferredType =
        RankedTensorType::get(projectedShape, inferredType.getElementType());
  }
  return inferredType;
}

RankedTensorType DispatchTensorLoadOp::inferResultType(
    IREE::Flow::DispatchTensorType sourceType,
    ArrayRef<OpFoldResult> mixedSizes) {
  auto shape = llvm::to_vector<4>(
      llvm::map_range(mixedSizes, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return attr.cast<IntegerAttr>().getInt();
        }
        return DispatchTensorType::kDynamicSize;
      }));
  return RankedTensorType::get(shape, sourceType.getElementType());
}

llvm::SmallBitVector DispatchTensorLoadOp::getDroppedDims() {
  return getDroppedDimsImpl(getType(), getMixedSizes());
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ValueRange sourceDynamicDims,
                                 ArrayRef<NamedAttribute> attributes) {
  SmallVector<OpFoldResult> offsets, strides, sizes;
  getDefaultOffsetSizeAndStrides(
      builder, source.getType().cast<IREE::Flow::DispatchTensorType>(),
      sourceDynamicDims, offsets, sizes, strides);
  build(builder, state, returnType, source, sourceDynamicDims, offsets, sizes,
        strides, attributes);
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ValueRange sourceDynamicDims,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;

  processMixedOperands(mixedOffsets, offsets, staticOffsets,
                       ShapedType::kDynamicStrideOrOffset);
  processMixedOperands(mixedSizes, sizes, staticSizes,
                       ShapedType::kDynamicSize);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamicStrideOrOffset);

  build(builder, state, returnType, source, sourceDynamicDims, offsets, sizes,
        strides, builder.getI64ArrayAttr(staticOffsets),
        builder.getI64ArrayAttr(staticSizes),
        builder.getI64ArrayAttr(staticStrides));
  state.addAttributes(attributes);
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 Value source, ValueRange sourceDynamicDims,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  auto returnType =
      inferResultType(source.getType().cast<DispatchTensorType>(), mixedSizes);
  build(builder, state, returnType, source, sourceDynamicDims, mixedOffsets,
        mixedSizes, mixedStrides);
}

LogicalResult DispatchTensorLoadOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto mixedSizes = getMixedSizes();
  SmallVector<Value> shape;
  if (!mixedSizes.empty()) {
    // Slicing out a tile; return the size sliced.
    shape = llvm::to_vector<6>(llvm::map_range(
        getMixedSizes(), [&](OpFoldResult valueOrAttr) -> Value {
          if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
            return b.create<arith::ConstantIndexOp>(
                getLoc(), attr.cast<IntegerAttr>().getInt());
          } else {
            return valueOrAttr.dyn_cast<Value>();
          }
        }));
  } else {
    // Result size matches the source size (no slicing).
    unsigned dynamicIdx = 0;
    for (int64_t dim : getType().getShape()) {
      if (dim == ShapedType::kDynamicSize) {
        shape.push_back(source_dims()[dynamicIdx++]);
      } else {
        shape.push_back(b.create<arith::ConstantIndexOp>(getLoc(), dim));
      }
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorStoreOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {target()}, target_dims()))) {
    return failure();
  }
  return success();
}

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ValueRange targetDynamicDims,
                                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  getDefaultOffsetSizeAndStrides(
      builder, target.getType().cast<IREE::Flow::DispatchTensorType>(),
      targetDynamicDims, offsets, sizes, strides);
  build(builder, state, value, target, targetDynamicDims, offsets, sizes,
        strides, attributes);
}

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ValueRange targetDynamicDims,
                                  ArrayRef<OpFoldResult> mixedOffsets,
                                  ArrayRef<OpFoldResult> mixedSizes,
                                  ArrayRef<OpFoldResult> mixedStrides,
                                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> offsets, sizes, strides;
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  processMixedOperands(mixedOffsets, offsets, staticOffsets,
                       ShapedType::kDynamicStrideOrOffset);
  processMixedOperands(mixedSizes, sizes, staticSizes,
                       ShapedType::kDynamicSize);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamicStrideOrOffset);

  build(builder, state, ArrayRef<Type>(), value, target, targetDynamicDims,
        offsets, sizes, strides, builder.getI64ArrayAttr(staticOffsets),
        builder.getI64ArrayAttr(staticSizes),
        builder.getI64ArrayAttr(staticStrides));
  state.addAttributes(attributes);
}

llvm::SmallBitVector DispatchTensorStoreOp::getDroppedDims() {
  return getDroppedDimsImpl(value().getType().cast<RankedTensorType>(),
                            getMixedSizes());
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

void DispatchWorkgroupsOp::build(OpBuilder &builder, OperationState &state,
                                 ValueRange workgroupCount,
                                 TypeRange resultTypes, ValueRange resultDims,
                                 ValueRange operands, ValueRange operandDims,
                                 ArrayRef<int64_t> tiedOperands,
                                 ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(workgroupCount);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(workgroupCount.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));

  auto *body = state.addRegion();
  assert(body->begin() == body->end());
  {
    OpBuilder::InsertionGuard g(builder);
    builder.createBlock(body);  // createBlock implicitly moves IP, RAII away...
  }

  llvm::BitVector operandAliases(llvm::size(operands), false);
  llvm::BitVector resultAliases(llvm::size(resultTypes), false);
  for (unsigned resultIndex = 0; resultIndex < tiedOperands.size();
       ++resultIndex) {
    int64_t tiedOperandIndex = tiedOperands[resultIndex];
    if (tiedOperandIndex != IREE::Util::TiedOpInterface::kUntiedIndex) {
      operandAliases[tiedOperandIndex] = true;
      resultAliases[resultIndex] = true;
    }
  }

  for (auto operand : llvm::enumerate(operands)) {
    Type type = operand.value().getType();
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchTensorType::get(operandAliases[operand.index()]
                                         ? TensorAccess::ReadWrite
                                         : TensorAccess::ReadOnly,
                                     tensorType);
    }
    body->addArgument(type, operand.value().getLoc());
  }
  for (auto resultType : llvm::enumerate(resultTypes)) {
    if (resultAliases[resultType.index()]) {
      // Already handled by an aliased operand.
      continue;
    }
    Type type = resultType.value();
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchTensorType::get(TensorAccess::WriteOnly, tensorType);
    }
    body->addArgument(type, state.location);
  }
  assert(std::next(body->begin()) == body->end());
}

static ParseResult parseDispatchWorkgroupBody(OpAsmParser &parser,
                                              TypeRange operandTypes,
                                              TypeRange resultTypes,
                                              Region &body) {
  SmallVector<OpAsmParser::UnresolvedOperand> regionArgs;
  SmallVector<Type> regionArgTypes;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Reserve entries in the lists.
      regionArgs.emplace_back();
      regionArgTypes.emplace_back();
      if (failed(parser.parseRegionArgument(regionArgs.back())) ||
          failed(parser.parseColonType(regionArgTypes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  return parser.parseRegion(body, regionArgs, regionArgTypes,
                            /*argLocations=*/{},
                            /*enableNameShadowing=*/true);
}

static void printDispatchWorkgroupBody(OpAsmPrinter &p, Operation *op,
                                       TypeRange operandTypes,
                                       TypeRange resultTypes, Region &body) {
  p << "(";
  interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p << arg;
    p << ": ";
    p << arg.getType();
  });
  p << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult DispatchWorkgroupsOp::verify() {
  Operation *op = getOperation();
  if (workgroup_count().empty()) {
    return op->emitOpError() << "at least one workgroup dimension is required";
  }

  if (failed(verifyOpDynamicDims(getOperation(), operands(), operand_dims())) ||
      failed(verifyOpDynamicDims(getOperation(), results(), result_dims()))) {
    return failure();
  }

  auto verifyIOType = [&](Type type) -> LogicalResult {
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      if (shapedType.getElementType().isIndex()) {
        return op->emitOpError() << "I/O type " << type
                                 << " is invalid: index types must not cross "
                                    "the dispatch boundary";
      }
    }
    return success();
  };
  for (auto type : getOperandTypes()) {
    if (failed(verifyIOType(type))) return failure();
  }
  for (auto type : getResultTypes()) {
    if (failed(verifyIOType(type))) return failure();
  }

  return success();
}

Operation::operand_range DispatchWorkgroupsOp::getClosureOperands() {
  return operands();
}

Operation::result_range DispatchWorkgroupsOp::getClosureResults() {
  return results();
}

// Inline operations that the dispatch region can handle natively.
static bool canDispatchRegionContainOp(Operation *op) {
  // Inline constant operations that are splat or small constants.
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    auto constantType = constantOp.getType();
    if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  return false;
}

bool DispatchWorkgroupsOp::canClosureContainOp(Operation *op) {
  return canDispatchRegionContainOp(op);
}

// Refines the tensor access from what is declared on |type| based on actual
// usage. We expect that the access was set correctly to begin with but today
// we sometimes specify things too wide.
static TensorAccess refineTensorAccess(Value value, DispatchTensorType type) {
  auto tensorAccess = type.getAccess();
  if (tensorAccess == TensorAccess::ReadWrite) {
    // If the argument is a result with `readwrite` access, return false if the
    // value is only written to. Check this by looking at the uses of the
    // argument being only the `target` of `flow.dispatch.tensor.store` ops.
    bool onlyWrites = true;
    for (OpOperand &uses : value.getUses()) {
      auto storeOp = dyn_cast<DispatchTensorStoreOp>(uses.getOwner());
      if (!(storeOp && storeOp.target() == uses.get())) {
        onlyWrites = false;
        break;
      }
    }
    if (onlyWrites) tensorAccess = TensorAccess::WriteOnly;
  }
  return tensorAccess;
}

IREE::Util::ValueAccess DispatchWorkgroupsOp::getOperandAccess(
    unsigned operandIndex) {
  BlockArgument arg = body().front().getArgument(operandIndex);
  if (auto tensorType = arg.getType().dyn_cast<DispatchTensorType>()) {
    auto tensorAccess = refineTensorAccess(arg, tensorType);
    return IREE::Util::ValueAccess(
        /*isRead=*/(tensorAccess == TensorAccess::ReadOnly) ||
            (tensorAccess == TensorAccess::ReadWrite),
        /*isWrite=*/(tensorAccess == TensorAccess::ReadWrite) ||
            (tensorAccess == TensorAccess::WriteOnly),
        /*isDiscard=*/(tensorAccess == TensorAccess::WriteOnly));
  } else {
    return IREE::Util::ValueAccess(/*isRead=*/!arg.use_empty(),
                                   /*isWrite=*/false,
                                   /*isDiscard=*/false);
  }
}

IREE::Util::ValueAccess DispatchWorkgroupsOp::getResultAccess(
    unsigned resultIndex) {
  unsigned startIndex = getBody()->getNumArguments() - getNumResults();
  BlockArgument arg = body().front().getArgument(startIndex + resultIndex);
  if (auto tensorType = arg.getType().dyn_cast<DispatchTensorType>()) {
    auto tensorAccess = refineTensorAccess(arg, tensorType);
    return IREE::Util::ValueAccess(
        /*isRead=*/(tensorAccess == TensorAccess::ReadOnly) ||
            (tensorAccess == TensorAccess::ReadWrite),
        /*isWrite=*/(tensorAccess == TensorAccess::ReadWrite) ||
            (tensorAccess == TensorAccess::WriteOnly),
        /*isDiscard=*/(tensorAccess == TensorAccess::WriteOnly));
  } else {
    return IREE::Util::ValueAccess(/*isRead=*/!arg.use_empty(),
                                   /*isWrite=*/false,
                                   /*isDiscard=*/false);
  }
}

// Recursively erases all users of |arg|.
// Assumes that it's possible to erase them all.
static void eraseArgUseTree(BlockArgument arg, PatternRewriter &rewriter) {
  SetVector<Operation *> deadOps;
  mlir::getForwardSlice(arg, &deadOps);
  for (auto deadOp : llvm::reverse(deadOps)) {
    rewriter.eraseOp(deadOp);
  }
}

IREE::Util::ClosureOpInterface
DispatchWorkgroupsOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  SmallVector<Type, 4> newResultTypes = llvm::to_vector<4>(getResultTypes());
  SmallVector<Value, 4> newResultDims = llvm::to_vector<4>(result_dims());
  SmallVector<Value, 4> newOperandsValues = llvm::to_vector<4>(operands());
  SmallVector<Value, 4> newOperandDims = llvm::to_vector<4>(operand_dims());
  IREE::Util::excludeClosureOperandsAndResults(
      newOperandsValues, newOperandDims, excludedOperandIndices, newResultTypes,
      newResultDims, excludedResultIndices);

  auto newTiedOperandIndices =
      llvm::to_vector<4>(getTiedResultOperandIndices());

  // TODO(benvanik): all this offset stuff is confusing and should be reworked.
  // We should probably have absolute indices and relative indices, or just one
  // or the other, and not be crossing the streams. The way things are offset
  // is the same as variadic ODS operands for consistency, but just like ODS
  // operands half of the code assumes its within a particular ODS operand and
  // half the code assumes it's within the flattened set of all Operation
  // operands.
  unsigned tiedOperandOffset = getTiedOperandsIndexAndLength().first;
  for (unsigned i = 0; i < newTiedOperandIndices.size(); ++i) {
    if (newTiedOperandIndices[i] != IREE::Util::TiedOpInterface::kUntiedIndex) {
      newTiedOperandIndices[i] -= tiedOperandOffset;
    }
  }

  // This need to happen *after* accounting for tied operand offset, given that
  // all excluded operand/result indices are relative ranges.
  IREE::Util::excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);

  auto newOp = rewriter.create<DispatchWorkgroupsOp>(
      getLoc(), workgroup_count(), newResultTypes, newResultDims,
      newOperandsValues, newOperandDims, newTiedOperandIndices,
      getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());

  // For dropped results, erase all the store-op uses. It is a pre-requisite
  // that the result can be dropped only if it is written within the dispatch
  // region op.
  unsigned baseResultIndex = operands().size();  // old index
  auto erasedArguments = llvm::to_vector<4>(excludedOperandIndices);
  for (unsigned i = baseResultIndex, e = newBody.getNumArguments(); i != e;
       ++i) {
    if (!is_contained(excludedResultIndices, i - baseResultIndex)) continue;
    auto arg = newBody.front().getArgument(i);
    eraseArgUseTree(arg, rewriter);
    erasedArguments.push_back(i);
  }
  newBody.front().eraseArguments(erasedArguments);

  return newOp;
}

std::pair<unsigned, unsigned>
DispatchWorkgroupsOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroup.*
//===----------------------------------------------------------------------===//

void DispatchWorkgroupRankOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "workgroup_rank");
}

static void getAsmResultNamesForDispatchWorkgroupInfoOp(
    StringRef prefix, const APInt &dimension, Value result,
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result, (prefix + std::to_string(dimension.getZExtValue())).str());
}

void DispatchWorkgroupIDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_id_", dimension(),
                                              result(), setNameFn);
}

void DispatchWorkgroupCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_count_", dimension(),
                                              result(), setNameFn);
}

void DispatchWorkgroupSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_size_", dimension(),
                                              result(), setNameFn);
}

LogicalResult verifyDispatchWorkgroupInfoOp(Operation *op, uint64_t dimension) {
  size_t dimCount = 0;
  if (auto dispatchOp = op->getParentOfType<DispatchWorkgroupsOp>()) {
    dimCount = dispatchOp.workgroup_count().size();
  }
  if (dimCount != 0 && (dimension < 0 || dimension >= dimCount)) {
    return op->emitOpError()
           << "dimension " << dimension
           << " out of bounds of dispatch dimensions; expected [0, "
           << (dimCount - 1) << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(OpBuilder &builder, OperationState &state,
                         StringRef name) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
}

LogicalResult ExecutableOp::verify() {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.entry
//===----------------------------------------------------------------------===//

void DispatchEntryOp::build(OpBuilder &builder, OperationState &state,
                            StringRef sym_name, FlatSymbolRefAttr function_ref,
                            IntegerAttr workgroup_rank) {
  build(builder, state, /*sym_visibility=*/nullptr,
        builder.getStringAttr(sym_name), function_ref, workgroup_rank);
}

ParseResult DispatchEntryOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  StringAttr visibilityAttr;
  if (failed(parseSymbolVisibility(parser, visibilityAttr))) {
    return failure();
  }

  FlatSymbolRefAttr functionRefAttr;
  if (failed(parser.parseAttribute(functionRefAttr, "function_ref",
                                   result.attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    StringAttr exportNameAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(exportNameAttr, "sym_name",
                                     result.attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    result.addAttribute("sym_name", parser.getBuilder().getStringAttr(
                                        functionRefAttr.getValue()));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return failure();
  }

  return success();
}

void DispatchEntryOp::print(OpAsmPrinter &p) {
  p << ' ';
  Operation *op = getOperation();
  printSymbolVisibility(p, op, op->getAttrOfType<StringAttr>("sym_visibility"));
  p << ' ';
  p.printSymbolName(function_ref());
  if (sym_name() != function_ref()) {
    p << " as(\"" << sym_name() << "\")";
  }
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(), /*elidedAttrs=*/{"function_ref", "sym_name"});
}

//===----------------------------------------------------------------------===//
// flow.dispatch
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &state,
                       DispatchEntryOp entryPoint, ValueRange workgroupCount,
                       TypeRange resultTypes, ValueRange resultDims,
                       ValueRange operands, ValueRange operandDims,
                       ArrayAttr tiedOperands,
                       ArrayRef<NamedAttribute> attributes) {
  StringRef executableOpSymName =
      entryPoint->getParentOp()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  state.addAttribute(
      "entry_point",
      SymbolRefAttr::get(builder.getContext(), executableOpSymName,
                         {SymbolRefAttr::get(entryPoint)}));

  state.addOperands(workgroupCount);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     tiedOperands);
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(workgroupCount.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
}

StringAttr DispatchOp::executable() { return entry_point().getRootReference(); }

FunctionType DispatchOp::getEntryPointType() {
  SmallVector<Type, 8> argTypes(operand_type_range{operands()});
  return FunctionType::get(getContext(), argTypes, getResultTypes());
}

LogicalResult DispatchOp::verify() {
  Operation *op = getOperation();
  if (workgroup_count().empty()) {
    return op->emitOpError() << "at least one workgroup dimension is required";
  }
  if (failed(verifyOpDynamicDims(op, operands(), operand_dims())) ||
      failed(verifyOpDynamicDims(op, results(), result_dims()))) {
    return failure();
  }
  return success();
}

std::pair<unsigned, unsigned> DispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);  // $operands
}

//===----------------------------------------------------------------------===//
// flow.tensor.clone
//===----------------------------------------------------------------------===//

LogicalResult TensorCloneOp::verify() {
  if (failed(
          verifyOpDynamicDims(getOperation(), {operand()}, operand_dims())) ||
      failed(verifyOpDynamicDims(getOperation(), {result()}, operand_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.empty
//===----------------------------------------------------------------------===//

LogicalResult TensorEmptyOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {result()}, result_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult TensorLoadOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {source()}, source_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.slice
//===----------------------------------------------------------------------===//

LogicalResult TensorSliceOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {source()}, source_dims())) ||
      failed(verifyOpDynamicDims(getOperation(), {result()}, result_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.splat
//===----------------------------------------------------------------------===//

LogicalResult TensorSplatOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {result()}, result_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult TensorStoreOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {target()}, target_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.tie_shape
//===----------------------------------------------------------------------===//

LogicalResult TensorTieShapeOp::verify() {
  if (failed(
          verifyOpDynamicDims(getOperation(), {operand()}, dynamic_dims()))) {
    return failure();
  }
  return success();
}

LogicalResult TensorTieShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<Value> shape;
  unsigned dynamicIdx = 0;
  auto tensorType = result().getType().cast<RankedTensorType>();
  for (int64_t dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamicSize) {
      shape.push_back(dynamic_dims()[dynamicIdx++]);
    } else {
      shape.push_back(b.create<arith::ConstantIndexOp>(getLoc(), dim));
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.reshape
//===----------------------------------------------------------------------===//

LogicalResult TensorReshapeOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {source()}, source_dims())) ||
      failed(
          verifyOpDynamicDims(getOperation(), {result()}, {result_dims()}))) {
    return failure();
  }
  return success();
}

Value TensorReshapeOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorReshapeOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorReshapeOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// flow.tensor.update
//===----------------------------------------------------------------------===//

void TensorUpdateOp::build(OpBuilder &builder, OperationState &state,
                           Value target, ValueRange startIndices,
                           Value update) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);
  auto updateDims =
      IREE::Util::buildDynamicDimsForValue(state.location, update, builder);
  build(builder, state, target.getType(), target, targetDims, startIndices,
        update, updateDims, builder.getIndexArrayAttr({0}));
}

LogicalResult TensorUpdateOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {update()}, update_dims())) ||
      failed(verifyOpDynamicDims(getOperation(), {target()}, target_dims()))) {
    return failure();
  }
  return success();
}

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> TensorUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// Public methods
//===----------------------------------------------------------------------===//

// Returns the offsets, sizes and strides to use when combining two operations
// that implement the `OffsetSizeAndStrideOpInterface`.
LogicalResult foldOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    const llvm::SmallBitVector &droppedProducerDims,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  SmallVector<OpFoldResult> consumerOffsets = consumer.getMixedOffsets();
  SmallVector<OpFoldResult> consumerSizes = consumer.getMixedSizes();
  SmallVector<OpFoldResult> consumerStrides = consumer.getMixedStrides();
  SmallVector<OpFoldResult> producerOffsets = producer.getMixedOffsets();
  SmallVector<OpFoldResult> producerSizes = producer.getMixedSizes();
  SmallVector<OpFoldResult> producerStrides = producer.getMixedStrides();

  combinedOffsets.resize(producerOffsets.size());
  combinedSizes.resize(producerOffsets.size());
  combinedStrides.resize(producerOffsets.size());
  unsigned consumerPos = 0;
  for (auto i : llvm::seq<unsigned>(0, producerOffsets.size())) {
    if (droppedProducerDims.test(i)) {
      // For dropped dims, get the values from the producer.
      combinedOffsets[i] = producerOffsets[i];
      combinedSizes[i] = producerSizes[i];
      combinedStrides[i] = producerStrides[i];
      continue;
    }
    SmallVector<Value> offsetSymbols, strideSymbols;
    // The combined offset is computed as
    //    producer_offset + consumer_offset * producer_strides.
    combinedOffsets[i] =
        getOpFoldResult(builder, loc,
                        add(mul(consumerOffsets[consumerPos],
                                producerStrides[i], offsetSymbols),
                            producerOffsets[i], offsetSymbols),
                        offsetSymbols);
    combinedSizes[i] = consumerSizes[consumerPos];
    // The combined stride is computed as
    //    consumer_stride * producer_stride.
    combinedStrides[i] = getOpFoldResult(
        builder, loc,
        mul(consumerStrides[consumerPos], producerStrides[i], strideSymbols),
        strideSymbols);
    consumerPos++;
  }
  return success();
}

/// Pattern to fold `flow.dispatch.tensor.load` -> `tensor.extract_slice`.
// TODO(ravishankarm): Eventually this should go in as a canonicalization at the
// Flow level.
struct FoldTensorLoadWithExtractSlice
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto dispatchTensorLoadOp =
        extractSliceOp.source()
            .getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!dispatchTensorLoadOp) return failure();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.extract_slice` (i.e. the producer) folds **into**
    // `flow.dispatch.tensor.load1 (i.e. the consumer).
    if (failed(foldOffsetsSizesAndStrides(
            rewriter, dispatchTensorLoadOp->getLoc(), dispatchTensorLoadOp,
            extractSliceOp, dispatchTensorLoadOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        extractSliceOp, extractSliceOp.getType(), dispatchTensorLoadOp.source(),
        dispatchTensorLoadOp.source_dims(), offsets, sizes, strides);
    return success();
  }
};

/// Pattern to fold `tensor.insert_slice` with `flow.dispatch.tensor.store`
/// oeprations.
// TODO(ravishankarm): Eventually this should go in as a canonicalization at the
// Flow level.
struct FoldInsertSliceWithTensorStoreOp
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorStoreOp dispatchTensorStoreOp,
      PatternRewriter &rewriter) const override {
    auto insertSliceOp =
        dispatchTensorStoreOp.value().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) return failure();

    // Check that the `dest` of the `tensor.insert_slice` and target of the
    // `flow.dispatch.tensor.store` are the same interface binding.
    Optional<BlockArgument> destBinding =
        getBindingArgument(insertSliceOp.dest());
    Optional<BlockArgument> targetBinding =
        getBindingArgument(dispatchTensorStoreOp.target());
    if (!destBinding || !targetBinding ||
        destBinding.getValue() != targetBinding.getValue()) {
      return failure();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.insert_slice` (i.e. the producer) folds **into**
    // `flow.dispatch.tensor.store` (i.e. the consumer).
    if (failed(foldOffsetsSizesAndStrides(
            rewriter, dispatchTensorStoreOp->getLoc(), dispatchTensorStoreOp,
            insertSliceOp, dispatchTensorStoreOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        dispatchTensorStoreOp, insertSliceOp.source(),
        dispatchTensorStoreOp.target(), dispatchTensorStoreOp.target_dims(),
        offsets, sizes, strides);
    return success();
  }
};

void populateFlowDispatchCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  DispatchTensorLoadOp::getCanonicalizationPatterns(results, context);
}

void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    mlir::RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .insert<FoldTensorLoadWithExtractSlice, FoldInsertSliceWithTensorStoreOp>(
          context);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc"  // IWYU pragma: keep
