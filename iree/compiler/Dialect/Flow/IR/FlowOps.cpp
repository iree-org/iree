// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of constant that can be inlined into a "
                   "dispatch region"),
    llvm::cl::init(256));

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

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

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

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ArrayRef<NamedAttribute> attributes) {
  build(builder, state, returnType, source, ArrayRef<Value>(),
        ArrayRef<Value>(), ArrayRef<Value>(), builder.getI64ArrayAttr({}),
        builder.getI64ArrayAttr({}), builder.getI64ArrayAttr({}));
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
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

  build(builder, state, returnType, source, offsets, sizes, strides,
        builder.getI64ArrayAttr(staticOffsets),
        builder.getI64ArrayAttr(staticSizes),
        builder.getI64ArrayAttr(staticStrides));
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 Value source,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  auto returnType =
      inferResultType(source.getType().cast<DispatchTensorType>(), mixedSizes);
  build(builder, state, returnType, source, mixedOffsets, mixedSizes,
        mixedStrides);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.store
//===----------------------------------------------------------------------===//

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ArrayRef<NamedAttribute> attributes) {
  build(builder, state, ArrayRef<Type>(), value, target, ArrayRef<Value>(),
        ArrayRef<Value>(), ArrayRef<Value>(), builder.getI64ArrayAttr({}),
        builder.getI64ArrayAttr({}), builder.getI64ArrayAttr({}));
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
    body->addArgument(type);
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
    body->addArgument(type);
  }
  assert(std::next(body->begin()) == body->end());
}

static ParseResult parseDispatchWorkgroupBody(OpAsmParser &parser,
                                              TypeRange operandTypes,
                                              TypeRange resultTypes,
                                              Region &body) {
  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
  SmallVector<Type, 16> regionArgTypes;
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
  p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static LogicalResult verifyDispatchWorkgroupsOp(DispatchWorkgroupsOp op) {
  if (op.workgroup_count().empty()) {
    return op.emitOpError() << "at least one workgroup dimension is required";
  }
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

Value DispatchWorkgroupsOp::buildOperandRankedShape(unsigned idx,
                                                    OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value DispatchWorkgroupsOp::buildResultRankedShape(unsigned idx,
                                                   OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
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
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    auto constantValueAttr = constantOp.getValue();
    auto constantType = constantOp.getType();
    if (constantValueAttr.isa<SplatElementsAttr>()) {
      return true;
    } else if (auto denseAttr =
                   constantValueAttr.dyn_cast<DenseElementsAttr>()) {
      // TODO(GH-4897): Non-splat constants seems to have an issue on the LLVM
      // side. Uncomment after that is fixed.
      auto shapedType = constantOp.getType().cast<ShapedType>();
      uint64_t estimatedByteLength =
          (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) /
          8;
      return denseAttr.isSplat() ||
             estimatedByteLength <= clInlineConstantByteLength;
    } else if (constantType.isIntOrIndexOrFloat()) {
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
  // Use old index when erasing ops.
  unsigned baseResultIndex = operands().size();
  // For dropped results, erase all the store-op uses. It is a pre-requisite
  // that the result can be dropped only if it is written within the dispatch
  // region op.
  auto erasedArguments = llvm::to_vector<4>(excludedOperandIndices);
  for (unsigned i = baseResultIndex, e = newBody.getNumArguments(); i != e;
       ++i) {
    if (!is_contained(excludedResultIndices, i - baseResultIndex)) continue;
    auto arg = newBody.front().getArgument(i);
    for (OpOperand &user : llvm::make_early_inc_range(arg.getUses())) {
      auto storeOp = dyn_cast<DispatchTensorStoreOp>(user.getOwner());
      if (storeOp && storeOp.target() == user.get()) {
        rewriter.eraseOp(storeOp);
      }
    }
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

template <typename T>
static LogicalResult verifyDispatchWorkgroupInfoOp(T op) {
  size_t dimCount = 0;
  if (auto dispatchOp = op->template getParentOfType<DispatchWorkgroupsOp>()) {
    dimCount = dispatchOp.workgroup_count().size();
  }
  uint64_t dimension = op.dimension().getZExtValue();
  if (dimCount != 0 && (dimension < 0 || dimension >= dimCount)) {
    return op.emitOpError()
           << "dimension " << dimension
           << " out of bounds of dispatch dimensions; expected [0, "
           << (dimCount - 1) << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.shape
//===----------------------------------------------------------------------===//

void DispatchShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // TODO(benvanik): since we know these are arguments, we could map them based
  // on index (so we get arg0_shape, ret0_shape, etc).
  setNameFn(result(), "shape");
}

LogicalResult DispatchShapeOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto dispatchTensorType = operands[0].getType().cast<DispatchTensorType>();
  auto shape = dispatchTensorType.getShape();
  auto rankedShapeType = Shape::RankedShapeType::get(shape, context);
  inferredReturnTypes.assign({rankedShapeType});
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

static LogicalResult verifyExecutableOp(ExecutableOp op) {
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

static ParseResult parseDispatchEntryOp(OpAsmParser &parser,
                                        OperationState *result) {
  StringAttr visibilityAttr;
  if (failed(parseSymbolVisibility(parser, visibilityAttr))) {
    return failure();
  }

  FlatSymbolRefAttr functionRefAttr;
  if (failed(parser.parseAttribute(functionRefAttr, "function_ref",
                                   result->attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    StringAttr exportNameAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(exportNameAttr, "sym_name",
                                     result->attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    result->addAttribute("sym_name", parser.getBuilder().getStringAttr(
                                         functionRefAttr.getValue()));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  return success();
}

static void printDispatchEntryOp(OpAsmPrinter &p, DispatchEntryOp op) {
  p << ' ';
  printSymbolVisibility(p, op, op->getAttrOfType<StringAttr>("sym_visibility"));
  p << ' ';
  p.printSymbolName(op.function_ref());
  if (op.sym_name() != op.function_ref()) {
    p << " as(\"" << op.sym_name() << "\")";
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

static LogicalResult verifyDispatchOp(DispatchOp op) {
  if (op.workgroup_count().empty()) {
    return op.emitOpError() << "at least one workgroup dimension is required";
  }
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

Value DispatchOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value DispatchOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
}

std::pair<unsigned, unsigned> DispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);  // $operands
}

//===----------------------------------------------------------------------===//
// flow.tensor.reshape
//===----------------------------------------------------------------------===//

Value TensorReshapeOp::buildOperandRankedShape(unsigned idx,
                                               OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorReshapeOp::buildResultRankedShape(unsigned idx,
                                              OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
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
// flow.tensor.*
//===----------------------------------------------------------------------===//

Value TensorLoadOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorLoadOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return {};
}

Value TensorStoreOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                         builder);
}

Value TensorStoreOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), target_dims(),
                                         builder);
}

Value TensorSplatOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return {};
}

Value TensorSplatOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
}

Value TensorCloneOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), operand(), operand_dims(),
                                         builder);
}

Value TensorCloneOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), operand_dims(),
                                         builder);
}

Value TensorSliceOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorSliceOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
}

//===----------------------------------------------------------------------===//
// flow.tensor.update
//===----------------------------------------------------------------------===//

void TensorUpdateOp::build(OpBuilder &builder, OperationState &state,
                           Value target, ValueRange startIndices,
                           Value update) {
  auto targetDims =
      Shape::buildOrFindDynamicDimsForValue(state.location, target, builder);
  auto updateDims =
      Shape::buildOrFindDynamicDimsForValue(state.location, update, builder);
  build(builder, state, target.getType(), target, targetDims, startIndices,
        update, updateDims, builder.getIndexArrayAttr({0}));
}

static LogicalResult verifyTensorUpdateOp(TensorUpdateOp op) {
  if (failed(verifyOpDynamicDims(op, {op.update()}, op.update_dims())) ||
      failed(verifyOpDynamicDims(op, {op.target()}, op.target_dims()))) {
    return failure();
  }
  return success();
}

Value TensorUpdateOp::buildOperandRankedShape(unsigned idx,
                                              OpBuilder &builder) {
  if (idx == 0) {
    return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                                 target_dims(), builder);
  } else {
    return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                                 update_dims(), builder);
  }
}

Value TensorUpdateOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                         builder);
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
// flow.ex.stream.fragment
//===----------------------------------------------------------------------===//

void ExStreamFragmentOp::build(OpBuilder &builder, OperationState &state,
                               TypeRange resultTypes, ValueRange resultDims,
                               ValueRange operands, ValueRange operandDims,
                               ArrayRef<int64_t> tiedOperands,
                               ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
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
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
  state.addRegion();
}

static LogicalResult verifyExStreamFragmentOp(ExStreamFragmentOp op) {
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

static ParseResult parseStreamFragmentBody(OpAsmParser &parser,
                                           TypeRange operandTypes,
                                           TypeRange resultTypes,
                                           ArrayAttr tiedOperands,
                                           Region &body) {
  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
  SmallVector<Type, 16> regionArgTypes;
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

  SmallVector<Type, 4> regionResultTypes;
  if (failed(parser.parseArrowTypeList(regionResultTypes))) return failure();

  if (regionArgs.size() != operandTypes.size()) {
    return parser.emitError(loc, "region operand list mismatch");
  }
  if (regionResultTypes.size() != resultTypes.size()) {
    return parser.emitError(loc, "region result list mismatch");
  }

  return parser.parseRegion(body, regionArgs, regionArgTypes,
                            /*enableNameShadowing=*/true);
}

static void printStreamFragmentBody(OpAsmPrinter &p, Operation *op,
                                    TypeRange operandTypes,
                                    TypeRange resultTypes,
                                    ArrayAttr tiedOperands, Region &body) {
  p << "(";
  llvm::interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p << arg;
    p << ": ";
    p << arg.getType();
  });
  p << ") -> ";
  if (resultTypes.size() != 1) p << "(";
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    p.printType(resultTypes[i]);
    if (i < resultTypes.size() - 1) p << ", ";
  }
  if (resultTypes.size() != 1) p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

Value ExStreamFragmentOp::buildOperandRankedShape(unsigned idx,
                                                  OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value ExStreamFragmentOp::buildResultRankedShape(unsigned idx,
                                                 OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
}

Operation::operand_range ExStreamFragmentOp::getClosureOperands() {
  return operands();
}

Operation::result_range ExStreamFragmentOp::getClosureResults() {
  return results();
}

bool ExStreamFragmentOp::canClosureContainOp(Operation *op) {
  // NOTE: we widen support on new stream ops only - the legacy path isn't worth
  // upgrading to support more.
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    return constantOp.getType().isIntOrIndexOrFloat();
  }
  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
    // Only allow loads of immutable variables to move into the stream.
    // As they are immutable it's always safe to do so as no synchronization at
    // the stream entry/exit boundary is required.
    //
    // Loads of mutable variables may sometimes be safe to move in as well
    // however that is best done when we have better cross-stream
    // synchronization support and can make those guarantees structurally.
    return loadOp.isGlobalImmutable();
  }
  return false;
}

IREE::Util::ValueAccess ExStreamFragmentOp::getOperandAccess(
    unsigned operandIndex) {
  return !isOperandTied(operandIndex) ? IREE::Util::ValueAccess::ReadOnly()
                                      : IREE::Util::ValueAccess::ReadWrite();
}

IREE::Util::ValueAccess ExStreamFragmentOp::getResultAccess(
    unsigned resultIndex) {
  return getTiedResultOperandIndex(resultIndex).hasValue()
             ? IREE::Util::ValueAccess::ReadWrite()
             : IREE::Util::ValueAccess::DiscardWrite();
}

IREE::Util::ClosureOpInterface
ExStreamFragmentOp::cloneReplacementExcludingOperandsAndResults(
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
  IREE::Util::excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);
  assert(getTiedOperandsIndexAndLength().first == 0 &&
         "operands must be the first ODS group");

  auto newOp = rewriter.create<ExStreamFragmentOp>(
      getLoc(), newResultTypes, newResultDims, newOperandsValues,
      newOperandDims, newTiedOperandIndices, getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  IREE::Util::eraseRegionResults(newBody, excludedResultIndices);
  newBody.front().eraseArguments(excludedOperandIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// Public methods
//===----------------------------------------------------------------------===//

void populateFlowDispatchCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  DispatchTensorLoadOp::getCanonicalizationPatterns(results, context);
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
