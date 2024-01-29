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
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::IREE::Flow {

//===----------------------------------------------------------------------===//
// Op utilities used within the Flow dialect
//===----------------------------------------------------------------------===//

// TODO(hanchung): Have a better fix. This is a fix for
// https://reviews.llvm.org/D124649
static void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                       ArrayRef<Type> types,
                       SmallVector<OpAsmParser::Argument> &args) {
  for (auto [operand, type] : llvm::zip_equal(operands, types)) {
    auto &arg = args.emplace_back();
    arg.ssaName = operand;
    arg.type = type;
  }
}

// Verifies that a dispatch |op|'s |workload| matches that of the |exportOp|.
static LogicalResult
verifyDispatchWorkload(Operation *op, IREE::Flow::ExecutableExportOp exportOp,
                       ValueRange workload) {
  // If the target has a workgroup count computation function we can verify that
  // the workload here matches what is expected.
  if (!exportOp.getWorkgroupCount().empty()) {
    auto &workgroupCount = exportOp.getWorkgroupCount();
    auto explicitArgs = llvm::make_filter_range(
        workgroupCount.getArgumentTypes(), [](Type type) {
          return !type.hasTrait<
              mlir::OpTrait::IREE::Util::ImplicitlyCaptured>();
        });
    if (llvm::range_size(explicitArgs) != workload.size()) {
      return op->emitOpError()
             << "workload mismatch; entry point expects "
             << llvm::range_size(explicitArgs)
             << " arguments but dispatch provides " << workload.size();
    }
    for (auto [index, expectedType, actualType] :
         llvm::enumerate(explicitArgs, workload.getTypes())) {
      if (expectedType != actualType) {
        return op->emitOpError()
               << "workload operand " << index << " type mismatch; expected "
               << expectedType << " but passed " << actualType;
      }
    }
  }
  return success();
}

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// of the dynamic dimensions in |values|.
static LogicalResult verifyOpDynamicDims(Operation *op, ValueRange values,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (auto shapedType = llvm::dyn_cast<ShapedType>(value.getType())) {
      requiredCount += shapedType.getNumDynamicDims();
    } else if (auto tensorType =
                   llvm::dyn_cast<DispatchTensorType>(value.getType())) {
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

static LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                          Operation *op,
                                          RankedTensorType expectedType) {
  switch (result) {
  case SliceVerificationResult::Success:
    return success();
  case SliceVerificationResult::RankTooLarge:
    return op->emitError("expected rank to be smaller or equal to ")
           << "the other rank. ";
  case SliceVerificationResult::SizeMismatch:
    return op->emitError("expected type to be ")
           << expectedType << " or a rank-reduced version. (size mismatch) ";
  case SliceVerificationResult::ElemTypeMismatch:
    return op->emitError("expected element type to be ")
           << expectedType.getElementType();
  default:
    llvm_unreachable("unexpected slicing op verification result");
  }
}

// Gets the dropped dimensions for `flow.dispatch.tensor.load/store`.
static llvm::SmallBitVector
getDroppedDimsImpl(RankedTensorType slicedObjectType,
                   ArrayRef<OpFoldResult> mixedSizes) {
  ArrayRef<int64_t> resultShape = slicedObjectType.getShape();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  size_t maxDroppedDims = mixedSizes.size() - resultShape.size();
  if (maxDroppedDims == 0) {
    return droppedDims;
  }
  unsigned shapePos = 0;
  int numSet = 0;
  for (const auto &size : llvm::enumerate(mixedSizes)) {
    std::optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || sizeVal.value() != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
    numSet++;
    if (numSet == maxDroppedDims) {
      break;
    }
  }
  return droppedDims;
}

/// Returns the `hal.interface.binding` a value comes from.
static std::optional<BlockArgument> getBindingArgument(Value v) {
  if (BlockArgument blockArg = llvm::dyn_cast<BlockArgument>(v)) {
    if (isa<IREE::Flow::DispatchWorkgroupsOp>(
            blockArg.getOwner()->getParentOp())) {
      return blockArg;
    }
    return std::nullopt;
  }
  Operation *definingOp = v.getDefiningOp();
  if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(definingOp)) {
    return getBindingArgument(loadOp.getSource());
  }
  return std::nullopt;
}

/// Returns `true` if the slice (described by the `offset`, `sizes` and
/// `strides`) spans the dispatch type.
static bool doesSliceSpanWholeTarget(
    IREE::Flow::DispatchTensorType dispatchType, ValueRange dispatchTypeDims,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) {
  // All offsets must be zero.
  if (!llvm::all_of(offsets, [](OpFoldResult ofr) {
        return isConstantIntValue(ofr, 0);
      })) {
    return false;
  }

  // All the sizes must match the entire target size.
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  auto targetType = dispatchType;
  if (staticSizes != targetType.getShape() ||
      llvm::any_of(llvm::zip_equal(dynamicSizes, dispatchTypeDims),
                   [](std::tuple<Value, Value> en) {
                     return std::get<0>(en) != std::get<1>(en);
                   })) {
    return false;
  }

  // All the strides must be 1.
  if (!llvm::all_of(strides, [](OpFoldResult ofr) {
        return isConstantIntValue(ofr, 1);
      })) {
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// custom<ShapedOperandList>($values, type($values), $value_dims)
//===----------------------------------------------------------------------===//
// %value : type{%dynamic_dims}, ...

static ParseResult parseShapedOperandList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    SmallVectorImpl<Type> &valueTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valueDims) {
  do {
    values.emplace_back();
    valueTypes.emplace_back();
    if (failed(parser.parseOperand(values.back())) ||
        failed(parser.parseColon()) ||
        failed(parser.parseType(valueTypes.back())))
      return failure();
    if (int64_t dynamicDimCount =
            cast<ShapedType>(valueTypes.back()).getNumDynamicDims()) {
      if (failed(parser.parseOperandList(valueDims, dynamicDimCount,
                                         AsmParser::Delimiter::Braces)))
        return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printShapedOperandList(OpAsmPrinter &p, Operation *op,
                                   ValueRange values, TypeRange valueTypes,
                                   ValueRange valueDims) {
  llvm::interleaveComma(llvm::zip_equal(values, valueTypes), p, [&](auto it) {
    auto [value, valueType] = it;
    p << value;
    p << " : ";
    p << valueType;
    if (int64_t dynamicDimCount =
            cast<ShapedType>(valueType).getNumDynamicDims()) {
      p << "{";
      llvm::interleaveComma(valueDims.take_front(dynamicDimCount), p);
      valueDims = valueDims.drop_front(dynamicDimCount);
      p << "}";
    }
  });
}

//===----------------------------------------------------------------------===//
// custom<WorkgroupCountRegion>($body)
//===----------------------------------------------------------------------===//

static ParseResult parseWorkgroupCountRegionWithoutKeyword(OpAsmParser &parser,
                                                           Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  if (failed(parser.parseArgumentList(args, AsmParser::Delimiter::Paren,
                                      /*allowType=*/true,
                                      /*allowAttrs=*/true))) {
    return failure();
  }

  // Return types must be 3 dimensions (workgroup count XYZ).
  SmallVector<Type> returnTypes;
  if (failed(parser.parseArrowTypeList(returnTypes))) {
    return failure();
  }
  if (returnTypes.size() != 3 ||
      !llvm::all_of(returnTypes, [](Type type) { return type.isIndex(); })) {
    return parser.emitError(parser.getCurrentLocation())
           << "workgroup count region must return the XYZ dimension counts";
  }

  // Parse region contents.
  if (failed(parser.parseRegion(body, args, /*enableNameShadowing=*/false))) {
    return failure();
  }

  // Verify the return types match.
  for (auto returnOp : body.getOps<IREE::Flow::ReturnOp>()) {
    for (auto [resultType, returnType] :
         llvm::zip_equal(returnTypes, returnOp.getOperandTypes())) {
      if (resultType != returnType) {
        return returnOp.emitOpError()
               << "operands do not match expected region return types";
      }
    }
  }

  return success();
}

static void printWorkgroupCountRegionWithoutKeyword(OpAsmPrinter &p,
                                                    Operation *op,
                                                    Region &body) {
  if (body.empty())
    return;
  p << "(";
  auto args = body.getArguments();
  for (unsigned i = 0; i < args.size(); ++i) {
    if (i > 0)
      p << ", ";
    p.printRegionArgument(args[i]);
  }
  p << ")";
  Type indexType = IndexType::get(body.getContext());
  p.printArrowTypeList(TypeRange{indexType, indexType, indexType});
  p << " ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

// TODO(benvanik): make these keywords required or consistent.

static ParseResult parseWorkgroupCountRegion(OpAsmParser &parser,
                                             Region &body) {
  if (failed(parser.parseOptionalKeyword("workgroups"))) {
    return success(); // Omitted.
  }
  return parseWorkgroupCountRegionWithoutKeyword(parser, body);
}

static void printWorkgroupCountRegion(OpAsmPrinter &p, Operation *op,
                                      Region &body) {
  if (body.empty())
    return;
  p << "workgroups";
  printWorkgroupCountRegionWithoutKeyword(p, op, body);
}

static ParseResult parseDispatchWorkgroupsCountRegion(OpAsmParser &parser,
                                                      Region &body) {
  if (failed(parser.parseOptionalKeyword("count"))) {
    return success(); // Omitted.
  }
  return parseWorkgroupCountRegionWithoutKeyword(parser, body);
}

static void printDispatchWorkgroupsCountRegion(OpAsmPrinter &p, Operation *op,
                                               Region &body) {
  if (body.empty())
    return;
  p << " count";
  printWorkgroupCountRegionWithoutKeyword(p, op, body);
}

//===----------------------------------------------------------------------===//
// custom<DispatchEntryPoints>($entry_points)
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchEntryPoints(OpAsmParser &parser,
                                            ArrayAttr &entryPointAttrsArray) {
  SmallVector<Attribute> entryPointAttrs;
  if (succeeded(parser.parseOptionalLBrace())) {
    do {
      SymbolRefAttr entryPointAttr;
      if (failed(parser.parseAttribute(entryPointAttr)))
        return failure();
      entryPointAttrs.push_back(entryPointAttr);
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRBrace()))
      return failure();
  } else {
    SymbolRefAttr entryPointAttr;
    if (failed(parser.parseAttribute(entryPointAttr)))
      return failure();
    entryPointAttrs.push_back(entryPointAttr);
  }
  entryPointAttrsArray = parser.getBuilder().getArrayAttr(entryPointAttrs);
  return success();
}

static void printDispatchEntryPoints(OpAsmPrinter &p, Operation *op,
                                     ArrayAttr entryPointAttrs) {
  if (entryPointAttrs.size() == 1) {
    p.printAttribute(entryPointAttrs.getValue().front());
  } else {
    p << '{';
    llvm::interleaveComma(entryPointAttrs, p.getStream(),
                          [&](Attribute attr) { p.printAttribute(attr); });
    p << '}';
  }
}

//===----------------------------------------------------------------------===//
// flow.dispatch.region
//===----------------------------------------------------------------------===//

static LogicalResult
verifyWorkgroupCountRegion(Operation *op, ValueRange workload, Region &region) {
  // Verify the workload operands match the expected capture args.
  if (workload.size() != region.getNumArguments()) {
    return op->emitOpError()
           << "workload operands and workgroup count args mismatch ("
           << workload.size() << " vs " << region.getNumArguments() << ")";
  }
  for (auto [index, values] :
       llvm::enumerate(llvm::zip_equal(workload, region.getArguments()))) {
    auto [workloadValue, capturedArg] = values;
    if (workloadValue.getType() != capturedArg.getType()) {
      return op->emitOpError()
             << "workload value " << index << " type mismatch; operand is "
             << workloadValue.getType() << " but region captures "
             << capturedArg.getType();
    }
  }

  // Verify the return ops all provide XYZ values.
  for (auto returnOp : region.getOps<IREE::Flow::ReturnOp>()) {
    if (returnOp.getNumOperands() != 3 ||
        !llvm::all_of(returnOp.getOperandTypes(),
                      [](Type type) { return type.isIndex(); })) {
      return returnOp.emitOpError() << "workgroup count region must return "
                                       "the XYZ dimension counts";
    }
  }

  return success();
}

LogicalResult DispatchRegionOp::verify() {
  // No block arguments.
  if (!getBody().getArguments().empty()) {
    return emitOpError() << "expected no block arguments";
  }

  // Verify terminator.
  SmallVector<Flow::ReturnOp> returnOps;
  for (Block &block : getBody()) {
    if (auto returnOp =
            dyn_cast_or_null<Flow::ReturnOp>(block.getTerminator())) {
      returnOps.push_back(returnOp);
    }
  }
  for (auto returnOp : returnOps) {
    for (const auto [resultType, returnType] :
         llvm::zip_equal(getResultTypes(), returnOp->getOperandTypes()))
      if (resultType != returnType) {
        return returnOp->emitOpError()
               << "operand types do not match with parent results";
      }
  }

  // Make sure that all returned values are ranked tensors.
  for (Type t : getResultTypes()) {
    if (!llvm::isa<RankedTensorType>(t)) {
      return emitOpError() << "only ranked tensor results are allowed";
    }
  }

  Region &workgroupCount = getWorkgroupCount();
  if (workgroupCount.empty()) {
    return success();
  }

  // If workgroup count region exists, check it has a single block.
  return verifyWorkgroupCountRegion(getOperation(), getWorkload(),
                                    getWorkgroupCount());
}

ParseResult DispatchRegionOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<Type> resultTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> allOperands;
  std::unique_ptr<Region> bodyRegion = std::make_unique<Region>();
  std::unique_ptr<Region> workloadCountRegion = std::make_unique<Region>();
  SmallVector<OpAsmParser::UnresolvedOperand> workloadOperands;
  SMLoc workloadOperandsLoc;
  (void)workloadOperandsLoc;
  if (succeeded(parser.parseOptionalLSquare())) {
    workloadOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(workloadOperands))
      return failure();
    if (parser.parseRSquare())
      return failure();
  }
  if (succeeded(parser.parseOptionalArrow())) {
    ParseResult typeListResult =
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
          if (parser.parseType(resultTypes.emplace_back()))
            return failure();
          auto shapedType = llvm::dyn_cast<ShapedType>(resultTypes.back());
          if (!shapedType)
            return success();
          if (shapedType.hasStaticShape())
            return success();
          SmallVector<OpAsmParser::UnresolvedOperand> dynamicDims;
          if (parser.parseOperandList(dynamicDims,
                                      shapedType.getNumDynamicDims(),
                                      OpAsmParser::Delimiter::Braces))
            return failure();
          allOperands.append(dynamicDims.begin(), dynamicDims.end());
          return success();
        });
    if (typeListResult)
      return failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  if (parser.parseRegion(*bodyRegion))
    return failure();

  if (parseDispatchWorkgroupsCountRegion(parser, *workloadCountRegion)) {
    return failure();
  }

  result.addRegion(std::move(bodyRegion));
  result.addRegion(std::move(workloadCountRegion));
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(allOperands.size()),
                           static_cast<int32_t>(workloadOperands.size())}));

  if (parser.resolveOperands(allOperands, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();
  if (parser.resolveOperands(workloadOperands,
                             parser.getBuilder().getIndexType(),
                             workloadOperandsLoc, result.operands)) {
    return failure();
  }

  result.addTypes(resultTypes);
  return success();
}

void DispatchRegionOp::print(OpAsmPrinter &p) {
  SmallVector<StringRef, 1> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  if (!getWorkload().empty()) {
    p << "[" << getWorkload() << "]";
  }
  p << " -> (";
  unsigned resultDimCounter = 0;
  for (const auto &it : llvm::enumerate(getResult().getTypes())) {
    Type type = it.value();
    p << type;
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
      if (!shapedType.hasStaticShape()) {
        p << "{";
        p << getResultDims().slice(resultDimCounter,
                                   shapedType.getNumDynamicDims());
        p << "}";
        resultDimCounter += shapedType.getNumDynamicDims();
      }
    }
    if (it.index() < getNumResults() - 1)
      p << ", ";
  }
  p << ")";
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << " ";

  bool printTerminator = true;
  if (auto *term =
          getBody().empty() ? nullptr : getBody().begin()->getTerminator()) {
    printTerminator = !term->getAttrDictionary().empty() ||
                      term->getNumOperands() != 0 || term->getNumResults() != 0;
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/printTerminator);

  printDispatchWorkgroupsCountRegion(p, *this, getWorkgroupCount());
}

ValueRange DispatchRegionOp::getResultDynamicDims(unsigned idx) {
  unsigned counter = 0;
  for (unsigned i = 0; i < idx; ++i)
    if (auto shapedType = llvm::dyn_cast<ShapedType>(getResultTypes()[i]))
      counter += shapedType.getNumDynamicDims();
  auto shapedType = llvm::dyn_cast<ShapedType>(getResultTypes()[idx]);
  return getResultDims().slice(counter,
                               shapedType ? shapedType.getNumDynamicDims() : 0);
}

/// Canonicalizes a DispatchRegionOp: Drop all unused results. Returns `true`
/// if the IR was modified.
bool dropUnusedDispatchRegionResults(RewriterBase &rewriter,
                                     Flow::DispatchRegionOp regionOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(regionOp);

  // Determine unused results and result types + dynamic dimensions of the new
  // op.
  llvm::DenseSet<unsigned> unusedResults;
  SmallVector<Type> resultTypes;
  SmallVector<Value> dynamicDims;
  unsigned dimOffset = 0;
  for (const auto &it : llvm::enumerate(regionOp.getResults())) {
    Type type = it.value().getType();
    auto shapedType = llvm::dyn_cast<ShapedType>(type);
    if (it.value().use_empty()) {
      unusedResults.insert(it.index());
    } else {
      resultTypes.push_back(type);
      ValueRange dims = regionOp.getResultDims().slice(
          dimOffset, shapedType.getNumDynamicDims());
      dynamicDims.append(dims.begin(), dims.end());
    }
    dimOffset += shapedType.getNumDynamicDims();
  }
  assert(dimOffset == regionOp.getResultDims().size() &&
         "expected that all dynamic dims were processed");

  // Nothing to do if all results are used.
  if (unusedResults.empty())
    return false;

  // Create new region and move over the body.
  auto newRegionOp = rewriter.create<Flow::DispatchRegionOp>(
      regionOp.getLoc(), resultTypes, dynamicDims, regionOp.getWorkload());
  newRegionOp.getBody().takeBody(regionOp.getBody());

  // Update terminator.
  auto returnOp =
      cast<Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
  SmallVector<Value> yieldedValues;
  for (const auto &it : llvm::enumerate(returnOp.getOperands()))
    if (!unusedResults.contains(it.index()))
      yieldedValues.push_back(it.value());
  rewriter.modifyOpInPlace(
      returnOp, [&]() { returnOp.getOperandsMutable().assign(yieldedValues); });

  // Replace all uses of the old op.
  SmallVector<Value> replacements(regionOp->getNumResults(), nullptr);
  unsigned resultCounter = 0;
  for (const auto &it : llvm::enumerate(regionOp.getResults()))
    if (!unusedResults.contains(it.index()))
      replacements[it.index()] = newRegionOp->getResult(resultCounter++);
  rewriter.replaceOp(regionOp, replacements);

  return true;
}

struct DispatchRegionDropUnusedResults
    : public OpRewritePattern<DispatchRegionOp> {
  using OpRewritePattern<DispatchRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchRegionOp regionOp,
                                PatternRewriter &rewriter) const final {
    return success(dropUnusedDispatchRegionResults(rewriter, regionOp));
  }
};

void DispatchRegionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<DispatchRegionDropUnusedResults>(context);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tie_shape
//===----------------------------------------------------------------------===//

LogicalResult DispatchTieShapeOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getOperand()},
                                 getDynamicDims()))) {
    return failure();
  }
  return success();
}

LogicalResult DispatchTieShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> shape;
  unsigned dynamicIdx = 0;
  auto tensorType =
      llvm::cast<IREE::Flow::DispatchTensorType>(getResult().getType());
  for (int64_t dim : tensorType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      shape.push_back(getDynamicDims()[dynamicIdx++]);
    } else {
      shape.push_back(b.getIndexAttr(dim));
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorLoadOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims()))) {
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
          llvm::cast<IntegerAttr>(valueOrAttr.dyn_cast<Attribute>()).getInt();
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

RankedTensorType
DispatchTensorLoadOp::inferResultType(IREE::Flow::DispatchTensorType sourceType,
                                      ArrayRef<OpFoldResult> mixedSizes) {
  auto shape =
      llvm::map_to_vector(mixedSizes, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          return llvm::cast<IntegerAttr>(attr).getInt();
        }
        return ShapedType::kDynamic;
      });
  return RankedTensorType::get(shape, sourceType.getBoundElementType());
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
      builder, llvm::cast<IREE::Flow::DispatchTensorType>(source.getType()),
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
                       ShapedType::kDynamic);
  processMixedOperands(mixedSizes, sizes, staticSizes, ShapedType::kDynamic);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamic);

  build(builder, state, returnType, source, sourceDynamicDims, offsets, sizes,
        strides, staticOffsets, staticSizes, staticStrides);
  state.addAttributes(attributes);
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 Value source, ValueRange sourceDynamicDims,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  auto returnType = inferResultType(
      llvm::cast<DispatchTensorType>(source.getType()), mixedSizes);
  build(builder, state, returnType, source, sourceDynamicDims, mixedOffsets,
        mixedSizes, mixedStrides);
}

LogicalResult DispatchTensorLoadOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto mixedSizes = getMixedSizes();
  SmallVector<OpFoldResult> shape;
  if (!mixedSizes.empty()) {
    // Slicing out a tile; return the size sliced.
    shape.reserve(mixedSizes.size());
    auto droppedDims = getDroppedDims();
    for (auto mixedSize : llvm::enumerate(mixedSizes)) {
      if (droppedDims.test(mixedSize.index())) {
        continue;
      }
      shape.push_back(mixedSize.value());
    }
  } else {
    // Result size matches the source size (no slicing).
    unsigned dynamicIdx = 0;
    for (int64_t dim : getType().getShape()) {
      if (ShapedType::isDynamic(dim)) {
        shape.push_back(getSourceDims()[dynamicIdx++]);
      } else {
        shape.push_back(b.getIndexAttr(dim));
      }
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

Value DispatchTensorLoadOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::std::optional<unsigned>
DispatchTensorLoadOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> DispatchTensorLoadOp::getTiedResultOperandIndices() {
  return {0}; // source
}

bool DispatchTensorLoadOp::isLoadOfWholeSource() {
  return doesSliceSpanWholeTarget(getSourceType(), getSourceDims(),
                                  getMixedOffsets(), getMixedSizes(),
                                  getMixedStrides());
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorStoreOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getTarget()},
                                 getTargetDims()))) {
    return failure();
  }

  // We only verify that the source tensor type is consistent with the type
  // inferred from the slice sizes.
  RankedTensorType sourceTensorType = getValue().getType();
  auto inferredType = RankedTensorType::get(getStaticSizes(),
                                            sourceTensorType.getElementType());
  SliceVerificationResult result =
      isRankReducedType(inferredType, sourceTensorType);
  return produceSliceErrorMsg(result, *this, inferredType);
}

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ValueRange targetDynamicDims,
                                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  getDefaultOffsetSizeAndStrides(
      builder, llvm::cast<IREE::Flow::DispatchTensorType>(target.getType()),
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
                       ShapedType::kDynamic);
  processMixedOperands(mixedSizes, sizes, staticSizes, ShapedType::kDynamic);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamic);

  build(builder, state, ArrayRef<Type>(), value, target, targetDynamicDims,
        offsets, sizes, strides, staticOffsets, staticSizes, staticStrides);
  state.addAttributes(attributes);
}

llvm::SmallBitVector DispatchTensorStoreOp::getDroppedDims() {
  return getDroppedDimsImpl(llvm::cast<RankedTensorType>(getValue().getType()),
                            getMixedSizes());
}

bool DispatchTensorStoreOp::isStoreToWholeTarget() {
  return doesSliceSpanWholeTarget(getTargetType(), getTargetDims(),
                                  getMixedOffsets(), getMixedSizes(),
                                  getMixedStrides());
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

void DispatchWorkgroupsOp::build(OpBuilder &builder, OperationState &state,
                                 ValueRange workload, TypeRange resultTypes,
                                 ValueRange resultDims, ValueRange arguments,
                                 ValueRange argumentDims,
                                 ArrayRef<int64_t> tiedOperands,
                                 ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(workload);
  state.addOperands(arguments);
  state.addOperands(argumentDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(workload.size()),
                         static_cast<int32_t>(arguments.size()),
                         static_cast<int32_t>(argumentDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));

  auto *workgroupBody = state.addRegion();
  assert(workgroupBody->begin() == workgroupBody->end());
  {
    // createBlock implicitly moves IP, RAII away...
    OpBuilder::InsertionGuard g(builder);
    builder.createBlock(workgroupBody);
  }

  llvm::BitVector operandAliases(llvm::size(arguments), false);
  llvm::BitVector resultAliases(llvm::size(resultTypes), false);
  for (unsigned resultIndex = 0; resultIndex < tiedOperands.size();
       ++resultIndex) {
    int64_t tiedOperandIndex = tiedOperands[resultIndex];
    if (tiedOperandIndex != IREE::Util::TiedOpInterface::kUntiedIndex) {
      operandAliases[tiedOperandIndex] = true;
      resultAliases[resultIndex] = true;
    }
  }
  for (auto operand : llvm::enumerate(arguments)) {
    Type type = operand.value().getType();
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
      type = DispatchTensorType::get(operandAliases[operand.index()]
                                         ? TensorAccess::ReadWrite
                                         : TensorAccess::ReadOnly,
                                     tensorType);
    }
    workgroupBody->addArgument(type, operand.value().getLoc());
  }
  for (auto resultType : llvm::enumerate(resultTypes)) {
    if (resultAliases[resultType.index()]) {
      // Already handled by an aliased operand.
      continue;
    }
    Type type = resultType.value();
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
      type = DispatchTensorType::get(TensorAccess::WriteOnly, tensorType);
    }
    workgroupBody->addArgument(type, state.location);
  }
  assert(std::next(workgroupBody->begin()) == workgroupBody->end());

  // NOTE: workgroup count region is empty; callers are expected to populate it
  // if they want it.
  state.addRegion();
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
      if (failed(parser.parseOperand(regionArgs.back(),
                                     /*allowResultNumber=*/false)) ||
          failed(parser.parseColonType(regionArgTypes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  SmallVector<OpAsmParser::Argument> args;
  createArgs(regionArgs, regionArgTypes, args);
  return parser.parseRegion(body, args, /*enableNameShadowing=*/true);
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

  if (failed(verifyOpDynamicDims(getOperation(), getArguments(),
                                 getArgumentDims())) ||
      failed(
          verifyOpDynamicDims(getOperation(), getResults(), getResultDims()))) {
    return failure();
  }

  auto verifyIOType = [&](Type type) -> LogicalResult {
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
      if (shapedType.getElementType().isIndex()) {
        return op->emitOpError() << "I/O type " << type
                                 << " is invalid: index types must not cross "
                                    "the dispatch boundary";
      }
    }
    return success();
  };
  for (auto type : getOperandTypes()) {
    if (failed(verifyIOType(type)))
      return failure();
  }
  for (auto type : getResultTypes()) {
    if (failed(verifyIOType(type)))
      return failure();
  }

  // Workgroup count region is optional.
  if (!getWorkgroupCount().empty()) {
    if (failed(verifyWorkgroupCountRegion(op, getWorkload(),
                                          getWorkgroupCount()))) {
      return failure();
    }
  }

  return success();
}

BlockArgument DispatchWorkgroupsOp::getOutputBlockArgument(unsigned idx) {
  std::optional<ArrayAttr> tiedOperands = getTiedOperands();
  if (!tiedOperands.has_value() || tiedOperands->empty()) {
    unsigned numInputs = getArguments().size();
    return getWorkgroupBody().getArguments().drop_front(numInputs)[idx];
  }

  // Some outputs are tied to inputs and share their block arguments.
  int64_t tiedOperand =
      llvm::cast<IntegerAttr>((*tiedOperands)[idx]).getValue().getSExtValue();
  if (tiedOperand != IREE::Util::TiedOpInterface::kUntiedIndex)
    // This output is tied to an input.
    return getInputBlockArgument(tiedOperand);

  unsigned nextOutArgIdx = getArguments().size();
  for (unsigned i = 0; i < idx; ++i)
    if (llvm::cast<IntegerAttr>((*tiedOperands)[i]).getValue().getSExtValue() ==
        IREE::Util::TiedOpInterface::kUntiedIndex)
      nextOutArgIdx++;
  return getWorkgroupBody().getArguments()[nextOutArgIdx];
}

SmallVector<BlockArgument> DispatchWorkgroupsOp::getOutputBlockArguments() {
  SmallVector<BlockArgument> result;
  for (unsigned i = 0; i < getNumResults(); ++i)
    result.push_back(getOutputBlockArgument(i));
  return result;
}

Operation::operand_range DispatchWorkgroupsOp::getClosureOperands() {
  return getArguments();
}

Operation::result_range DispatchWorkgroupsOp::getClosureResults() {
  return getResults();
}

bool DispatchWorkgroupsOp::canClosureContainOp(Operation *op) {
  // For now we only allow constants; we could bring other ops across the
  // boundary though if we want (particularly metadata ops).
  // Note that the closure optimization may still filter out the constant op if
  // it's not configured to inline constants of certain types/sizes.
  // TODO(#12233): this should just be isa<ConstantOp> but today we need to
  // ensure that we don't mess with tensors after dispatch region formation due
  // to requirements around tensor access checking.
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    auto constantType = constantOp.getType();
    if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  return false;
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
    bool hasReads = false;
    bool hasWrites = false;
    for (OpOperand &uses : value.getUses()) {
      TypeSwitch<Operation *>(uses.getOwner())
          .Case<DispatchTensorLoadOp>([&](auto loadOp) { hasReads = true; })
          .Case<DispatchTensorStoreOp>([&](auto storeOp) { hasWrites = true; })
          .Default([&](auto op) {
            // Treat unknown ops conservatively as read/write.
            hasReads = true;
            hasWrites = true;
          });
    }
    if (hasReads && !hasWrites)
      tensorAccess = TensorAccess::ReadOnly;
    if (!hasReads && hasWrites)
      tensorAccess = TensorAccess::WriteOnly;
  }
  return tensorAccess;
}

IREE::Util::ValueAccess
DispatchWorkgroupsOp::getOperandAccess(unsigned operandIndex) {
  BlockArgument arg = getWorkgroupBody().front().getArgument(operandIndex);
  if (auto tensorType = llvm::dyn_cast<DispatchTensorType>(arg.getType())) {
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

IREE::Util::ValueAccess
DispatchWorkgroupsOp::getResultAccess(unsigned resultIndex) {
  unsigned startIndex = getWorkgroupBody().getNumArguments() - getNumResults();
  BlockArgument arg =
      getWorkgroupBody().front().getArgument(startIndex + resultIndex);
  if (auto tensorType = llvm::dyn_cast<DispatchTensorType>(arg.getType())) {
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
  SmallVector<Type> newResultTypes = llvm::to_vector(getResultTypes());
  SmallVector<Value> newResultDims = llvm::to_vector(getResultDims());
  SmallVector<Value> newArguments = llvm::to_vector(getArguments());
  SmallVector<Value> newArgumentDims = llvm::to_vector(getArgumentDims());
  IREE::Util::excludeClosureOperandsAndResults(
      newArguments, newArgumentDims, excludedOperandIndices, newResultTypes,
      newResultDims, excludedResultIndices);

  auto newTiedOperandIndices = llvm::to_vector(getTiedResultOperandIndices());

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
      getLoc(), getWorkload(), newResultTypes, newResultDims, newArguments,
      newArgumentDims, newTiedOperandIndices, getOperation()->getAttrs());
  newOp->setDialectAttrs(getOperation()->getDialectAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());

  // Copy the workgroup_count region.
  auto &workgroupCountRegion = getWorkgroupCount();
  if (!workgroupCountRegion.empty()) {
    auto &newWorkgroupCountRegion = newOp.getWorkgroupCount();
    newWorkgroupCountRegion.takeBody(workgroupCountRegion);
  }

  // For dropped results, erase all the store-op uses. It is a pre-requisite
  // that the result can be dropped only if it is written within the dispatch
  // region op.
  unsigned baseResultIndex = getArguments().size(); // old index
  auto erasedArguments = llvm::to_vector(excludedOperandIndices);
  for (unsigned i = baseResultIndex, e = newBody.getNumArguments(); i != e;
       ++i) {
    if (!is_contained(excludedResultIndices, i - baseResultIndex))
      continue;
    auto arg = newBody.front().getArgument(i);
    eraseArgUseTree(arg, rewriter);
    erasedArguments.push_back(i);
  }
  auto &block = newBody.front();
  BitVector eraseIndices(block.getNumArguments());
  for (auto i : erasedArguments)
    eraseIndices.set(i);
  block.eraseArguments(eraseIndices);

  return newOp;
}

std::pair<unsigned, unsigned>
DispatchWorkgroupsOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);
}

SmallVector<int64_t> DispatchWorkgroupsOp::getTiedOperandsAsIntegerList() {
  ArrayAttr attr = getTiedOperandsAttr();
  if (!attr)
    return {};
  return llvm::map_to_vector(attr, [](Attribute intAttr) {
    return llvm::cast<IntegerAttr>(intAttr).getInt();
  });
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroup.*
//===----------------------------------------------------------------------===//

static void getAsmResultNamesForDispatchWorkgroupInfoOp(
    StringRef prefix, const APInt &dimension, Value result,
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result, (prefix + std::to_string(dimension.getZExtValue())).str());
}

void DispatchWorkgroupIDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_id_", getDimension(),
                                              getResult(), setNameFn);
}

void DispatchWorkgroupCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp(
      "workgroup_count_", getDimension(), getResult(), setNameFn);
}

void DispatchWorkgroupSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_size_", getDimension(),
                                              getResult(), setNameFn);
}

LogicalResult verifyDispatchWorkgroupInfoOp(Operation *op, uint64_t dimension) {
  if (dimension < 0 || dimension >= 3) {
    return op->emitOpError()
           << "dimension " << dimension
           << " out of bounds of dispatch dimensions; expected [0, 3)";
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
// flow.executable.export
//===----------------------------------------------------------------------===//

void ExecutableExportOp::build(OpBuilder &builder, OperationState &state,
                               StringRef sym_name,
                               FlatSymbolRefAttr function_ref) {
  build(builder, state, /*sym_visibility=*/nullptr,
        builder.getStringAttr(sym_name), function_ref);
}

LogicalResult ExecutableExportOp::verify() {
  // Workgroup count region is optional.
  if (!getWorkgroupCount().empty()) {
    // Verify the return ops all provide XYZ values.
    for (auto returnOp : getWorkgroupCount().getOps<IREE::Flow::ReturnOp>()) {
      if (returnOp.getNumOperands() != 3 ||
          !llvm::all_of(returnOp.getOperandTypes(),
                        [](Type type) { return type.isIndex(); })) {
        return returnOp.emitOpError()
               << "workgroup count region must return the XYZ dimension counts";
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &state,
                       ExecutableExportOp exportOp, ValueRange workload,
                       TypeRange resultTypes, ValueRange resultDims,
                       ValueRange operands, ValueRange operandDims,
                       ArrayAttr tiedOperands,
                       ArrayRef<NamedAttribute> attributes) {
  StringRef executableOpSymName =
      exportOp->getParentOp()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  auto entryPoint =
      SymbolRefAttr::get(builder.getContext(), executableOpSymName,
                         {SymbolRefAttr::get(exportOp)});
  build(builder, state, entryPoint, workload, resultTypes, resultDims, operands,
        operandDims, tiedOperands, attributes);
}

void DispatchOp::build(OpBuilder &builder, OperationState &state,
                       SymbolRefAttr entryPoint, ValueRange workload,
                       TypeRange resultTypes, ValueRange resultDims,
                       ValueRange operands, ValueRange operandDims,
                       ArrayAttr tiedOperands,
                       ArrayRef<NamedAttribute> attributes) {
  state.addAttribute("entry_points", builder.getArrayAttr(entryPoint));
  state.addOperands(workload);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     tiedOperands);
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(workload.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
}

FunctionType DispatchOp::getEntryPointType() {
  SmallVector<Type, 8> argTypes(operand_type_range{getArguments()});
  return FunctionType::get(getContext(), argTypes, getResultTypes());
}

std::string DispatchOp::getEntryPointName() {
  // Pick the first entry point we have. The common case is we only have one
  // but frontends may provide multiple variants - they're all likely the
  // same name but with slight differences and enough for a user to know what's
  // happening.
  auto anyEntryPoint = *getEntryPointRefs().begin();
  std::string entryPointName =
      anyEntryPoint.getRootReference().getValue().str();
  for (FlatSymbolRefAttr nestedRef : anyEntryPoint.getNestedReferences()) {
    entryPointName = (entryPointName + "::" + nestedRef.getValue()).str();
  }
  return entryPointName;
}

std::pair<unsigned, unsigned> DispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1); // $operands
}

LogicalResult DispatchOp::verify() {
  Operation *op = getOperation();

  if (getEntryPoints().empty()) {
    return op->emitOpError("at least one entry point reference is required");
  }

  if (failed(verifyOpDynamicDims(op, getArguments(), getArgumentDims())) ||
      failed(verifyOpDynamicDims(op, getResults(), getResultDims()))) {
    return failure();
  }

  return success();
}

LogicalResult DispatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto entryPointRefs = getEntryPointRefs();
  if (entryPointRefs.empty()) {
    return emitOpError() << "at least one entry point must be defined";
  }
  for (auto entryPointAttr : entryPointRefs) {
    auto exportOp =
        symbolTable.lookupNearestSymbolFrom<IREE::Flow::ExecutableExportOp>(
            op, entryPointAttr);
    if (!exportOp) {
      // TODO(benvanik): there are a lot of tests that are assuming this is not
      // verified. We'll need to go add dummy executables for all of them. Today
      // we just bail on the verifier if the symbol isn't found.
      //
      // Should be:
      //   return op->emitOpError() << "undefined entry point: " <<
      //   getEntryPoint();
      return success();
    }

    // Verify that the workload parameters captured match the target export.
    if (failed(verifyDispatchWorkload(op, exportOp, getWorkload()))) {
      return failure();
    }

    // TODO(benvanik): verify that the target function has matching operands.
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.func
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<int64_t> tiedOperands,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs,
                      ArrayRef<DictionaryAttr> resAttrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type,
                builder.getIndexArrayAttr(tiedOperands), attrs, argAttrs,
                resAttrs);
  return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayAttr tiedOperands,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs,
                   ArrayRef<DictionaryAttr> resAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(SymbolTable::getVisibilityAttrName(),
                     builder.getStringAttr("private"));
  state.addAttribute("function_type", TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     tiedOperands);
  state.addRegion();
  if (!argAttrs.empty() || !resAttrs.empty()) {
    assert(type.getNumInputs() == argAttrs.size());
    assert(type.getNumResults() == resAttrs.size());
    function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, resAttrs, builder.getStringAttr("arg_attrs"),
        builder.getStringAttr("res_attrs"));
  }
}

//===----------------------------------------------------------------------===//
// flow.call
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder &builder, OperationState &state,
                   SymbolRefAttr callee, TypeRange resultTypes,
                   ValueRange resultDims, ValueRange arguments,
                   ValueRange argumentDims, ArrayAttr tiedOperands,
                   ArrayRef<NamedAttribute> attributes) {
  state.addAttribute("callee", callee);
  state.addTypes(resultTypes);
  state.addOperands(arguments);
  state.addOperands(argumentDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     tiedOperands);
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(arguments.size()),
                         static_cast<int32_t>(argumentDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
}

FunctionType CallOp::getCalleeType() {
  auto argumentTypes = llvm::map_to_vector(
      getArgOperands(), [](Value arg) { return arg.getType(); });
  return FunctionType::get(getContext(), argumentTypes, getResultTypes());
}

std::pair<unsigned, unsigned> CallOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(0); // $arguments
}

LogicalResult CallOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyOpDynamicDims(op, getArguments(), getArgumentDims())) ||
      failed(verifyOpDynamicDims(op, getResults(), getResultDims()))) {
    return failure();
  }
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();

  auto calleeOp = symbolTable.lookupNearestSymbolFrom<IREE::Flow::FuncOp>(
      op, getCalleeAttr());
  if (!calleeOp) {
    return op->emitOpError() << "undefined external call: " << getCallee();
  }

  auto expectedType = getCalleeType();
  auto calleeType = calleeOp.getFunctionType();
  if (calleeType != expectedType) {
    return emitOpError("function type mismatch; expected ")
           << expectedType << " but callee is " << calleeType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.tie_shape
//===----------------------------------------------------------------------===//

LogicalResult TensorTieShapeOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getOperand()},
                                 getDynamicDims()))) {
    return failure();
  }
  return success();
}

LogicalResult TensorTieShapeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> shape;
  unsigned dynamicIdx = 0;
  auto tensorType = llvm::cast<RankedTensorType>(getResult().getType());
  for (int64_t dim : tensorType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      shape.push_back(getDynamicDims()[dynamicIdx++]);
    } else {
      shape.push_back(b.getIndexAttr(dim));
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.reshape
//===----------------------------------------------------------------------===//

LogicalResult TensorReshapeOp::verify() {
  // The element types don't need to match but the bit widths need to.
  auto sourceType = llvm::cast<ShapedType>(getSource().getType());
  auto resultType = llvm::cast<ShapedType>(getResult().getType());
  if (IREE::Util::getTypeBitWidth(sourceType.getElementType()) !=
      IREE::Util::getTypeBitWidth(resultType.getElementType())) {
    return emitOpError() << "element bit widths must match";
  }

  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 {getResultDims()}))) {
    return failure();
  }

  return success();
}

Value TensorReshapeOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::std::optional<unsigned>
TensorReshapeOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> TensorReshapeOp::getTiedResultOperandIndices() {
  return {0}; // source
}

//===----------------------------------------------------------------------===//
// flow.tensor.bitcast
//===----------------------------------------------------------------------===//

LogicalResult TensorBitCastOp::verify() {
  // The element types don't need to match, we can just check the requisite
  // number of dynamic dims.
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 {getResultDims()}))) {
    return failure();
  }

  return success();
}

Value TensorBitCastOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::std::optional<unsigned>
TensorBitCastOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> TensorBitCastOp::getTiedResultOperandIndices() {
  return {0}; // source
}

//===----------------------------------------------------------------------===//
// flow.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult TensorLoadOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult TensorStoreOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getTarget()},
                                 getTargetDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.alloca
//===----------------------------------------------------------------------===//

LogicalResult TensorAllocaOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 getResultDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.empty
//===----------------------------------------------------------------------===//

LogicalResult TensorEmptyOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 getResultDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.splat
//===----------------------------------------------------------------------===//

LogicalResult TensorSplatOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 getResultDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.clone
//===----------------------------------------------------------------------===//

LogicalResult TensorCloneOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getOperand()},
                                 getArgumentDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 getArgumentDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.tensor.slice
//===----------------------------------------------------------------------===//

LogicalResult TensorSliceOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 getResultDims()))) {
    return failure();
  }
  return success();
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
        update, updateDims);
}

LogicalResult TensorUpdateOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getUpdate()},
                                 getUpdateDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getTarget()},
                                 getTargetDims()))) {
    return failure();
  }
  return success();
}

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::std::optional<unsigned>
TensorUpdateOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0}; // target
}

//===----------------------------------------------------------------------===//
// flow.tensor.trace
//===----------------------------------------------------------------------===//

LogicalResult TensorTraceOp::verify() {
  TensorTraceOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getValues(), op.getValueDims()))) {
    return failure();
  }
  return success();
}

ValueRange TensorTraceOp::getOperandDynamicDims(unsigned idx) {
  auto valueDims = getValueDims();
  for (unsigned i = 0; i <= idx; ++i) {
    auto valueType = cast<ShapedType>(getValues()[i].getType());
    int64_t dynamicDimCount = valueType.getNumDynamicDims();
    if (i == idx) {
      return valueDims.take_front(dynamicDimCount);
    }
    valueDims = valueDims.drop_front(dynamicDimCount);
  }
  return ValueRange{};
}

ValueRange TensorTraceOp::getResultDynamicDims(unsigned idx) {
  return ValueRange{};
}

//===----------------------------------------------------------------------===//
// Public methods
//===----------------------------------------------------------------------===//

/// Pattern to fold `flow.dispatch.tensor.load` -> `tensor.extract_slice`.
// TODO(ravishankarm): Eventually this should go in as a canonicalization at the
// Flow level.
struct FoldTensorLoadWithExtractSlice
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto dispatchTensorLoadOp =
        extractSliceOp.getSource()
            .getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!dispatchTensorLoadOp)
      return failure();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.extract_slice` (i.e. the producer) folds **into**
    // `flow.dispatch.tensor.load1 (i.e. the consumer).
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, dispatchTensorLoadOp->getLoc(), dispatchTensorLoadOp,
            extractSliceOp, dispatchTensorLoadOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        extractSliceOp, extractSliceOp.getType(),
        dispatchTensorLoadOp.getSource(), dispatchTensorLoadOp.getSourceDims(),
        offsets, sizes, strides);
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

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp dispatchTensorStoreOp,
                  PatternRewriter &rewriter) const override {
    auto insertSliceOp =
        dispatchTensorStoreOp.getValue().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp)
      return failure();

    // Check that the `dest` of the `tensor.insert_slice` and target of the
    // `flow.dispatch.tensor.store` are the same interface binding.
    std::optional<BlockArgument> destBinding =
        getBindingArgument(insertSliceOp.getDest());
    std::optional<BlockArgument> targetBinding =
        getBindingArgument(dispatchTensorStoreOp.getTarget());
    if (!destBinding || !targetBinding ||
        destBinding.value() != targetBinding.value()) {
      return failure();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.insert_slice` (i.e. the producer) folds **into**
    // `flow.dispatch.tensor.store` (i.e. the consumer).
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, dispatchTensorStoreOp->getLoc(), dispatchTensorStoreOp,
            insertSliceOp, dispatchTensorStoreOp.getDroppedDims(), offsets,
            sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        dispatchTensorStoreOp, insertSliceOp.getSource(),
        dispatchTensorStoreOp.getTarget(),
        dispatchTensorStoreOp.getTargetDims(), offsets, sizes, strides);
    return success();
  }
};

void populateFlowDispatchCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  DispatchTensorLoadOp::getCanonicalizationPatterns(results, context);
}

void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .insert<FoldTensorLoadWithExtractSlice, FoldInsertSliceWithTensorStoreOp>(
          context);
}

//===----------------------------------------------------------------------===//
// flow.channel.count
//===----------------------------------------------------------------------===//

void ChannelCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel_count");
}

//===----------------------------------------------------------------------===//
// flow.channel.default
//===----------------------------------------------------------------------===//

void ChannelDefaultOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "default_channel");
}

//===----------------------------------------------------------------------===//
// flow.channel.split
//===----------------------------------------------------------------------===//

void ChannelSplitOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel");
}

//===----------------------------------------------------------------------===//
// flow.channel.rank
//===----------------------------------------------------------------------===//

void ChannelRankOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel_rank");
}

//===----------------------------------------------------------------------===//
// flow.collective.all_gather
//===----------------------------------------------------------------------===//

Value CollectiveAllGatherOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::std::optional<unsigned>
CollectiveAllGatherOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> CollectiveAllGatherOp::getTiedResultOperandIndices() {
  return {0}; // target
}

void CollectiveAllGatherOp::build(OpBuilder &builder, OperationState &state,
                                  CollectiveElementTypeAttr elementType,
                                  Value target, Value source, Value channel) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);
  build(builder, state, elementType, target, targetDims, source, channel,
        builder.getIndexArrayAttr({0}));
}

//===----------------------------------------------------------------------===//
// flow.collective.all_reduce
//===----------------------------------------------------------------------===//

Value CollectiveAllReduceOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::std::optional<unsigned>
CollectiveAllReduceOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> CollectiveAllReduceOp::getTiedResultOperandIndices() {
  return {0}; // target
}

void CollectiveAllReduceOp::build(OpBuilder &builder, OperationState &state,
                                  CollectiveReductionOpAttr reductionOp,
                                  CollectiveElementTypeAttr elementType,
                                  Value target, Value source, Value channel) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);
  build(builder, state, reductionOp, elementType, target, targetDims, source,
        channel, builder.getIndexArrayAttr({0}));
}

//===----------------------------------------------------------------------===//
// flow.collective.all_to_all
//===----------------------------------------------------------------------===//

Value CollectiveAllToAllOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned>
CollectiveAllToAllOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> CollectiveAllToAllOp::getTiedResultOperandIndices() {
  return {0}; // target
}

void CollectiveAllToAllOp::build(OpBuilder &builder, OperationState &state,
                                 CollectiveElementTypeAttr elementType,
                                 Value target, Value source, Value channel) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);

  build(builder, state, elementType, target, targetDims, source, channel,
        builder.getIndexArrayAttr({0}));
}

//===----------------------------------------------------------------------===//
// flow.collective.reduce_scatter
//===----------------------------------------------------------------------===//

Value CollectiveReduceScatterOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::std::optional<unsigned>
CollectiveReduceScatterOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> CollectiveReduceScatterOp::getTiedResultOperandIndices() {
  return {0}; // target
}

void CollectiveReduceScatterOp::build(OpBuilder &builder, OperationState &state,
                                      CollectiveReductionOpAttr reductionOp,
                                      CollectiveElementTypeAttr elementType,
                                      Value target, Value source,
                                      Value channel) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);
  build(builder, state, reductionOp, elementType, target, targetDims, source,
        channel, builder.getIndexArrayAttr({0}));
}

//===----------------------------------------------------------------------===//
// flow.collective.send_recv
//===----------------------------------------------------------------------===//

Value CollectiveSendRecvOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned>
CollectiveSendRecvOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // target
}

SmallVector<int64_t> CollectiveSendRecvOp::getTiedResultOperandIndices() {
  return {0}; // target
}

void CollectiveSendRecvOp::build(OpBuilder &builder, OperationState &state,
                                 CollectiveElementTypeAttr elementType,
                                 Value target, Value source, Value channel,
                                 Value send, Value recv) {
  auto targetDims =
      IREE::Util::buildDynamicDimsForValue(state.location, target, builder);

  build(builder, state, elementType, target, targetDims, source, channel, send,
        recv, builder.getIndexArrayAttr({0}));
}

} // namespace mlir::iree_compiler::IREE::Flow

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc" // IWYU pragma: keep
