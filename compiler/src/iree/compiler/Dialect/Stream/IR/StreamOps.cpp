// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"

#include "iree/compiler/Dialect/Stream/Builtins/Builtins.h"
#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/ADTExtras.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Op utilities used within the stream dialect
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
static LogicalResult verifyDispatchWorkload(
    Operation *op, IREE::Stream::ExecutableExportOp exportOp,
    ValueRange workload) {
  // If the target has a workgroup count computation function we can verify that
  // the workload here matches what is expected.
  if (!exportOp.getWorkgroupCount().empty()) {
    auto &workgroupCount = exportOp.getWorkgroupCount();
    if (workgroupCount.getNumArguments() != workload.size()) {
      return op->emitOpError()
             << "workload mismatch; entry point expects "
             << workgroupCount.getNumArguments()
             << " arguments but dispatch provides " << workload.size();
    }
    for (auto [idx, expectedType, actualType] : enumerate_zip_equal(
             workgroupCount.getArgumentTypes(), workload.getTypes())) {
      if (expectedType != actualType) {
        return op->emitOpError()
               << "workload operand " << idx << " type mismatch; expected "
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

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// the dynamic dimensions in |type|.
static LogicalResult verifyOpDynamicDims(Operation *op, TypeRange types,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto type : types) {
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      requiredCount += shapedType.getNumDynamicDims();
    }
  }
  if (dynamicDims.size() != requiredCount) {
    return op->emitOpError()
           << "type set has " << requiredCount
           << " dynamic dimensions but only " << dynamicDims.size()
           << " dimension values are attached";
  }
  return success();
}

// Verifies that |sizes| contains the appropriate number of sizes for all of the
// sized types in |values|.
static LogicalResult verifyOpValueSizes(Operation *op, ValueRange values,
                                        ValueRange sizes) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (value.getType().isa<IREE::Util::SizeAwareTypeInterface>()) {
      ++requiredCount;
    }
  }
  if (sizes.size() != requiredCount) {
    return op->emitOpError() << "value set has " << requiredCount
                             << " dynamic dimensions but only " << sizes.size()
                             << " dimension values are attached";
  }
  return success();
}

// Verifies that all !stream.resources used within |region| are captured by
// the entry arguments to the region.
static LogicalResult verifyAllResourcesCaptured(Region &region) {
  SetVector<Value> availableResources;
  for (auto arg : region.front().getArguments()) {
    availableResources.insert(arg);
  }
  for (auto &op : region.front()) {
    for (auto result : op.getResults()) {
      availableResources.insert(result);
    }
    for (auto operand : op.getOperands()) {
      if (!operand) continue;
      if (!operand.getType().isa<IREE::Stream::ResourceType>()) continue;
      if (!availableResources.contains(operand)) {
        return op.emitOpError() << "used resource not listed in explicit "
                                   "captures (or produced internally)";
      }
    }
  }
  return success();
}

// Verifies that escaping !stream.resources have the sizes when they are
// yielded match the sizes declared on the parent op. This information is
// redundant but keeps analysis local and agnostic to the parent op structure
// which is useful for when we outline things.
static LogicalResult verifyEscapingResources(Region &region,
                                             ResultRange results,
                                             ValueRange resultSizes) {
  // Ensure yielded resources match the signature.
  for (auto yieldOp : region.getOps<IREE::Stream::YieldOp>()) {
    if (results.size() != yieldOp.getResourceOperands().size()) {
      return yieldOp.emitOpError()
             << "yield result count mismatch with parent op";
    }
    for (auto [outerValue, innerValue] :
         llvm::zip_equal(results, yieldOp.getResourceOperands())) {
      if (outerValue.getType() != innerValue.getType()) {
        return yieldOp.emitOpError()
               << "result type mismatch: expected " << outerValue.getType()
               << " but got " << innerValue.getType();
      }
    }
    for (auto [outerSize, innerSize] :
         llvm::zip_equal(resultSizes, yieldOp.getResourceOperandSizes())) {
      if (outerSize != innerSize) {
        return yieldOp.emitOpError() << "result size mismatch";
      }
    }
  }
  return success();
}

// Computes the value access bits starting from |rootValue|.
// Traverses the IR graph along tied ops but does not handle branches.
static IREE::Util::ValueAccess computeValueAccess(Value rootValue) {
  IREE::Util::ValueAccess access;
  DenseSet<Value> processedValues;
  SmallVector<Value> worklist;
  auto enqueueValue = [&](Value value) {
    if (processedValues.contains(value)) return;
    processedValues.insert(value);
    worklist.push_back(value);
  };
  enqueueValue(rootValue);
  while (!worklist.empty()) {
    Value value = worklist.back();
    worklist.pop_back();

    // Walk up the definition chain.
    if (auto definingOp = value.getDefiningOp()) {
      // Value is produced within the region and thus written.
      access.isWrite = true;
      if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
        access.isRead = true;
        auto operand = tiedOp.getTiedResultOperand(value);
        if (operand) {
          // Value is tied back to another value; continue analyzing past it.
          enqueueValue(operand);
        } else {
          // Value contents are fully produced by this op.
          access.isDiscard = true;
        }
      } else if (isa<IREE::Stream::SubviewEffectOpInterface>(definingOp)) {
        // TODO(benvanik): actually query; for now assume *.
        access.isRead = true;
        access.isWrite = true;
      } else {
        // Value contents are fully produced by this op.
        access.isDiscard = true;
      }
    }

    // Walk down the use chain.
    for (auto user : value.getUsers()) {
      // Used by an op.
      access.isRead = true;
      if (auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user)) {
        auto tiedIndices = tiedOp.getTiedResultOperandIndices();
        for (int64_t tiedIndex : tiedIndices) {
          if (tiedIndex == IREE::Util::TiedOpInterface::kUntiedIndex) continue;
          auto operand = user->getOperand(tiedIndex);
          if (operand == value) {
            // Tied operand.
            access.isRead = true;
            access.isWrite = true;
            enqueueValue(operand);
          }
        }
      } else if (isa<IREE::Stream::SubviewEffectOpInterface>(user)) {
        // TODO(benvanik): actually query; for now assume *.
        access.isRead = true;
        access.isWrite = true;
      }
    }
  }
  return access;
}

static void eraseStreamRegionResults(Region &region,
                                     ArrayRef<unsigned> excludedResultIndices) {
  for (auto &block : region.getBlocks()) {
    auto yieldOp = dyn_cast<IREE::Stream::YieldOp>(block.getTerminator());
    if (!yieldOp) continue;
    llvm::SmallVector<Value, 4> newOperands;
    for (auto i : llvm::reverse(excludedResultIndices)) {
      yieldOp.getResourceOperandsMutable().erase(i);
      yieldOp.getResourceOperandSizesMutable().erase(i);
    }
  }
}

//===----------------------------------------------------------------------===//
// custom<ResourceRegion>($operands, type($operands), $operand_sizes,
//                        type($results), $result_sizes,
//                        $tied_operands, $body)
//===----------------------------------------------------------------------===//

static ParseResult parseResourceRegion(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandSizes,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultSizes,
    ArrayAttr &tiedOperands, Region &body) {
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Reserve entries in the lists.
      operands.emplace_back();
      operandTypes.emplace_back();
      operandSizes.emplace_back();
      regionArgs.emplace_back();
      if (failed(parser.parseOperand(operands.back())) ||
          failed(parser.parseKeyword("as")) ||
          failed(parser.parseOperand(regionArgs.back(),
                                     /*allowResultNumber=*/false)) ||
          failed(parser.parseColon()) ||
          failed(parseSizeAwareType(parser, operandTypes.back(),
                                    operandSizes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (succeeded(parser.parseOptionalRParen())) {
        // -> ()
      } else if (failed(parseShapedResultList(parser, operands, operandTypes,
                                              operandSizes, resultTypes,
                                              resultSizes, tiedOperands)) ||
                 failed(parser.parseRParen())) {
        return failure();
      }
    } else {
      if (failed(parseShapedResultList(parser, operands, operandTypes,
                                       operandSizes, resultTypes, resultSizes,
                                       tiedOperands))) {
        return failure();
      }
    }
  }

  SmallVector<OpAsmParser::Argument> args;
  createArgs(regionArgs, operandTypes, args);
  return parser.parseRegion(body, args);
}

static void printResourceRegion(OpAsmPrinter &p, Operation *op,
                                ValueRange operands, TypeRange operandTypes,
                                ValueRange operandSizes, TypeRange resultTypes,
                                ValueRange resultSizes, ArrayAttr tiedOperands,
                                Region &body) {
  p << "(";
  llvm::interleaveComma(
      llvm::zip_equal(operands, body.getArguments()), p, [&](auto it) {
        auto operand = std::get<0>(it);
        auto arg = std::get<1>(it);
        p << operand;
        p << " as ";
        p << arg;
        p << ": ";
        p << arg.getType();
        if (arg.getType().template isa<IREE::Util::SizeAwareTypeInterface>()) {
          p << "{" << operandSizes.front() << "}";
          operandSizes = operandSizes.drop_front(1);
        }
      });
  p << ")";
  if (!resultTypes.empty()) {
    p << " -> ";
    if (resultTypes.size() != 1) p << "(";
    printShapedResultList(p, op, operands, operandTypes, operandSizes,
                          resultTypes, resultSizes, tiedOperands);
    if (resultTypes.size() != 1) p << ")";
  }
  p << " ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// custom<ExplicitResourceRegion>($operands, type($operands), $operand_sizes,
//                                $body)
//===----------------------------------------------------------------------===//

static ParseResult parseExplicitResourceRegion(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandSizes,
    Region &body) {
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Reserve entries in the lists.
      operands.emplace_back();
      operandTypes.emplace_back();
      operandSizes.emplace_back();
      regionArgs.emplace_back();
      if (failed(parser.parseOperand(operands.back())) ||
          failed(parser.parseKeyword("as")) ||
          failed(parser.parseOperand(regionArgs.back(),
                                     /*allowResultNumber=*/false)) ||
          failed(parser.parseColon()) ||
          failed(parseSizeAwareType(parser, operandTypes.back(),
                                    operandSizes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  SmallVector<OpAsmParser::Argument> args;
  createArgs(regionArgs, operandTypes, args);
  if (failed(parser.parseRegion(body, args))) {
    return failure();
  }
  // HACK: I can't figure out how to make this work with the default parsing -
  // it doesn't call this like it should.
  IREE::Stream::CmdExecuteOp::ensureTerminator(
      body, parser.getBuilder(),
      parser.getEncodedSourceLoc(parser.getCurrentLocation()));
  return success();
}

static void printExplicitResourceRegion(OpAsmPrinter &p, Operation *op,
                                        ValueRange operands,
                                        TypeRange operandTypes,
                                        ValueRange operandSizes, Region &body) {
  p << "(";
  llvm::interleaveComma(
      llvm::zip_equal(operands, body.getArguments()), p, [&](auto it) {
        auto operand = std::get<0>(it);
        auto arg = std::get<1>(it);
        p << operand;
        p << " as ";
        p << arg;
        p << ": ";
        p << arg.getType();
        if (arg.getType().template isa<IREE::Util::SizeAwareTypeInterface>()) {
          p << "{" << operandSizes.front() << "}";
          operandSizes = operandSizes.drop_front(1);
        }
      });
  p << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// custom<PackSliceRanges>($lifetime_intervals,
//                         $dynamic_slice_sizes,
//                         type($packed_offsets))
//===----------------------------------------------------------------------===//

static ParseResult parsePackSliceRanges(
    OpAsmParser &parser, ArrayAttr &lifetimeIntervals,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSliceSizes,
    SmallVectorImpl<Type> &packedOffsetTypes) {
  auto indexType = parser.getBuilder().getIndexType();
  SmallVector<Attribute> lifetimeRangeValues;
  do {
    if (failed(parser.parseOptionalLSquare())) break;
    IntegerAttr lifetimeStart;
    IntegerAttr lifetimeEnd;
    OpAsmParser::UnresolvedOperand dynamicSliceSize;
    if (failed(parser.parseAttribute(lifetimeStart, indexType)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseAttribute(lifetimeEnd, indexType)) ||
        failed(parser.parseRSquare()) || failed(parser.parseEqual()) ||
        failed(parser.parseOperand(dynamicSliceSize))) {
      return failure();
    }
    lifetimeRangeValues.push_back(lifetimeStart);
    lifetimeRangeValues.push_back(lifetimeEnd);
    dynamicSliceSizes.push_back(dynamicSliceSize);
    packedOffsetTypes.push_back(indexType);
  } while (succeeded(parser.parseOptionalComma()));
  lifetimeIntervals = parser.getBuilder().getArrayAttr(lifetimeRangeValues);
  return success();
}

static void printPackSliceRanges(OpAsmPrinter &p, Operation *op,
                                 ArrayAttr lifetimeIntervals,
                                 ValueRange dynamicSliceSizes,
                                 TypeRange packedOffsetTypes) {
  if (packedOffsetTypes.empty()) return;
  for (unsigned i = 0; i < packedOffsetTypes.size(); ++i) {
    auto lifetimeStart = lifetimeIntervals[i * 2];
    auto lifetimeEnd = lifetimeIntervals[i * 2 + 1];
    auto sliceSize = dynamicSliceSizes[i];
    p.printNewline();
    p << "  [";
    p.printAttributeWithoutType(lifetimeStart);
    p << ", ";
    p.printAttributeWithoutType(lifetimeEnd);
    p << "] = ";
    p.printOperand(sliceSize);
    if (i < packedOffsetTypes.size() - 1) p << ",";
  }
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// custom<ConstantValueList>(type($results),
//                           $result_sizes,
//                           $values)
//===----------------------------------------------------------------------===//
//   !stream.resource<constant>{%sz} = #value,
//   !stream.resource<constant>{%sz} = #value

static ParseResult parseConstantValueList(
    OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultSizes,
    ArrayAttr &values) {
  SmallVector<Attribute> valueAttrs;
  do {
    Type resultType;
    OpAsmParser::UnresolvedOperand resultSize;
    Attribute valueAttr;
    if (failed(parseSizeAwareType(parser, resultType, resultSize)) ||
        failed(parser.parseEqual()) ||
        failed(parser.parseAttribute(valueAttr))) {
      return failure();
    }
    resultTypes.push_back(resultType);
    resultSizes.push_back(resultSize);
    valueAttrs.push_back(valueAttr);
  } while (succeeded(parser.parseOptionalComma()));
  values = parser.getBuilder().getArrayAttr(valueAttrs);
  return success();
}

static void printConstantValueList(OpAsmPrinter &p, Operation *op,
                                   TypeRange resultTypes,
                                   ValueRange resultSizes, ArrayAttr values) {
  if (resultTypes.empty()) return;
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    p.printNewline();
    p << "  ";
    printSizeAwareType(p, op, resultTypes[i], resultSizes[i]);
    p << " = ";
    p.printAttribute(values[i]);
    if (i < resultTypes.size() - 1) p << ",";
  }
}

//===----------------------------------------------------------------------===//
// custom<WorkgroupCountRegion>($body)
//===----------------------------------------------------------------------===//

static ParseResult parseWorkgroupCountRegion(OpAsmParser &parser,
                                             Region &body) {
  if (failed(parser.parseOptionalKeyword("workgroups"))) {
    // Omitted.
    return success();
  }

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
  for (auto returnOp : body.getOps<IREE::Stream::ReturnOp>()) {
    for (auto [returnType, operandType] :
         llvm::zip_equal(returnTypes, returnOp.getOperandTypes())) {
      if (returnType != operandType) {
        return returnOp.emitOpError()
               << "operands do not match expected region return types";
      }
    }
  }

  return success();
}

static void printWorkgroupCountRegion(OpAsmPrinter &p, Operation *op,
                                      Region &body) {
  if (body.empty()) return;
  p << "workgroups(";
  auto args = body.getArguments();
  for (unsigned i = 0; i < args.size(); ++i) {
    if (i > 0) p << ", ";
    p.printRegionArgument(args[i]);
  }
  p << ")";
  Type indexType = IndexType::get(body.getContext());
  p.printArrowTypeList(TypeRange{indexType, indexType, indexType});
  p << " ";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// stream.resource.alloc
//===----------------------------------------------------------------------===//

LogicalResult ResourceAllocOp::verify() {
  ResourceAllocOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResults(), op.getStorageSizes()))) {
    return failure();
  }

  // All allocated resources must have the same lifetime.
  auto anyType = op.getResults().front().getType();
  for (auto type : op.getResultTypes()) {
    if (type != anyType) {
      return op.emitError()
             << "all allocated resources must have the same lifetime";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.map
//===----------------------------------------------------------------------===//

LogicalResult ResourceMapOp::verify() {
  ResourceMapOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.try_map
//===----------------------------------------------------------------------===//

LogicalResult ResourceTryMapOp::verify() {
  ResourceTryMapOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.load
//===----------------------------------------------------------------------===//

LogicalResult ResourceLoadOp::verify() {
  ResourceLoadOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.store
//===----------------------------------------------------------------------===//

LogicalResult ResourceStoreOp::verify() {
  ResourceStoreOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.pack
//===----------------------------------------------------------------------===//

void ResourcePackOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // TODO(benvanik): figure out if we can get the names to coalesce when there
  // are multiple results. Ideally we'd have `%total_length, %offsets:123` but
  // unfortunately all get splatted out and create 10k+ char lines that are a
  // pain to read.
  // setNameFn(total_length(), "total_length");
  // for (auto packedOffset : llvm::enumerate(packed_offsets())) {
  // setNameFn(packedOffset.value(),
  //           "offset" + std::to_string(packedOffset.index()));
  // }
}

LogicalResult ResourcePackOp::verify() {
  ResourcePackOp op = *this;
  size_t sliceCount = op.getPackedOffsets().size();
  if (op.getLifetimeIntervals().size() != sliceCount * 2) {
    return op.emitOpError() << "requires a [start, end] range for each slice";
  }
  if (op.getDynamicSliceSizes().size() != sliceCount) {
    return op.emitOpError() << "requires a size for each slice";
  }
  return success();
}

SmallVector<ResourcePackOp::Slice> ResourcePackOp::getSlices() {
  auto intervalPairs = getLifetimeIntervals().getValue();
  auto sizes = getDynamicSliceSizes();
  auto offsets = getPackedOffsets();
  SmallVector<ResourcePackOp::Slice> slices(offsets.size());
  for (size_t i = 0; i < offsets.size(); ++i) {
    int64_t start = intervalPairs[i * 2 + 0].cast<IntegerAttr>().getInt();
    int64_t end = intervalPairs[i * 2 + 1].cast<IntegerAttr>().getInt();
    slices[i] = {start, end, sizes[i], offsets[i]};
  }
  return slices;
}

//===----------------------------------------------------------------------===//
// stream.resource.constants
//===----------------------------------------------------------------------===//

LogicalResult ResourceConstantsOp::verify() {
  ResourceConstantsOp op = *this;
  size_t count = op.getResults().size();
  if (op.getResultSizes().size() != count || op.getValues().size() != count) {
    return op.emitOpError() << "mismatched constant/result counts";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.subview
//===----------------------------------------------------------------------===//

LogicalResult ResourceSubviewOp::verify() {
  ResourceSubviewOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

bool ResourceSubviewOp::isMetadata() { return true; }

Value ResourceSubviewOp::getViewSource() { return getSource(); }

Value ResourceSubviewOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::llvm::Optional<unsigned> ResourceSubviewOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> ResourceSubviewOp::getTiedResultOperandIndices() {
  return {0};  // source
}

// static
IREE::Stream::ResourceSubviewOp ResourceSubviewOp::findSubviewOp(Value value) {
  while (value) {
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) {
      // Defined as a block argument - stop walk.
      break;
    } else if (isa<IREE::Stream::TimelineOpInterface>(definingOp)) {
      // Don't traverse timeline ops as time travel isn't possible (yet).
      break;
    } else if (auto subviewOp =
                   dyn_cast<IREE::Stream::ResourceSubviewOp>(definingOp)) {
      // Found!
      return subviewOp;
    } else if (auto tiedOp =
                   dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
      // Continue walking up through the tied operand.
      value = tiedOp.getTiedResultOperand(value);
    } else {
      break;
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// stream.tensor.import
//===----------------------------------------------------------------------===//

LogicalResult TensorImportOp::verify() {
  TensorImportOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

Value TensorImportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::llvm::Optional<unsigned> TensorImportOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorImportOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// stream.tensor.export
//===----------------------------------------------------------------------===//

LogicalResult TensorExportOp::verify() {
  TensorExportOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getSourceEncoding(),
                                 op.getSourceEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize()))) {
    return failure();
  }
  return success();
}

Value TensorExportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::llvm::Optional<unsigned> TensorExportOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorExportOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// stream.tensor.sizeof
//===----------------------------------------------------------------------===//

LogicalResult TensorSizeOfOp::verify() {
  TensorSizeOfOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getEncoding(), op.getEncodingDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.empty
//===----------------------------------------------------------------------===//

void TensorEmptyOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "empty");
}

LogicalResult TensorEmptyOp::verify() {
  TensorEmptyOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.constant
//===----------------------------------------------------------------------===//

void TensorConstantOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "cst");
}

LogicalResult TensorConstantOp::verify() {
  TensorConstantOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.splat
//===----------------------------------------------------------------------===//

LogicalResult TensorSplatOp::verify() {
  TensorSplatOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.clone
//===----------------------------------------------------------------------===//

LogicalResult TensorCloneOp::verify() {
  TensorCloneOp op = *this;
  // Clones can't change encodings but they can change shape information.
  auto sourceEncoding = op.getSourceEncoding().cast<RankedTensorType>();
  auto resultEncoding = op.getResultEncoding().cast<RankedTensorType>();
  if (sourceEncoding.getEncoding() != resultEncoding.getEncoding()) {
    return op.emitOpError() << "clones changing tensor encoding from "
                            << sourceEncoding.getEncoding() << " to "
                            << resultEncoding.getEncoding() << "; not allowed";
  }
  if (failed(verifyOpDynamicDims(op, op.getSourceEncoding(),
                                 op.getSourceEncodingDims())) ||
      failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.slice
//===----------------------------------------------------------------------===//

LogicalResult TensorSliceOp::verify() {
  TensorSliceOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getSourceEncoding(),
                                 op.getSourceEncodingDims())) ||
      failed(verifyOpDynamicDims(op, op.getResultEncoding(),
                                 op.getResultEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  auto sourceType = op.getSourceEncoding().cast<ShapedType>();
  if (op.getStartIndices().size() != sourceType.getRank() ||
      op.getLengths().size() != sourceType.getRank()) {
    return op.emitOpError() << "start_indices/lengths rank mismatch";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.update
//===----------------------------------------------------------------------===//

LogicalResult TensorUpdateOp::verify() {
  TensorUpdateOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getUpdateEncoding(),
                                 op.getUpdateEncodingDims())) ||
      failed(verifyOpDynamicDims(op, op.getTargetEncoding(),
                                 op.getTargetEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getUpdate(), op.getUpdateSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> TensorUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.tensor.fill
//===----------------------------------------------------------------------===//

LogicalResult TensorFillOp::verify() {
  TensorFillOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getTargetEncoding(),
                                 op.getTargetEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value TensorFillOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> TensorFillOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorFillOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult TensorLoadOp::verify() {
  TensorLoadOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getSourceEncoding(),
                                 op.getSourceEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize()))) {
    return failure();
  }
  auto sourceType = op.getSourceEncoding().cast<ShapedType>();
  if (op.getIndices().size() != sourceType.getRank()) {
    return op.emitOpError() << "indices rank mismatch";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult TensorStoreOp::verify() {
  TensorStoreOp op = *this;
  if (failed(verifyOpDynamicDims(op, op.getTargetEncoding(),
                                 op.getTargetEncodingDims())) ||
      failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  auto targetType = op.getTargetEncoding().cast<ShapedType>();
  if (op.getIndices().size() != targetType.getRank()) {
    return op.emitOpError() << "indices rank mismatch";
  }
  return success();
}

Value TensorStoreOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> TensorStoreOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorStoreOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.builtin.* utilities
//===----------------------------------------------------------------------===//

// Merges a builtin module from iree/compiler/Dialect/Stream/Builtins/*.mlir
// into the user module; this allows for host functions and multiple
// executables.
//
// Fails if there's a name conflict; we have a __ prefix and things outside the
// compiler shouldn't use it.
static LogicalResult mergeBuiltinModuleSource(Location loc, StringRef fileName,
                                              Operation *targetOp,
                                              OpBuilder &targetBuilder) {
  // Find the file in the embedded data.
  const iree_file_toc_t *toc = iree_compiler_Stream_Builtins_create();
  const iree_file_toc_t *file = nullptr;
  for (size_t i = 0; i < iree_compiler_Stream_Builtins_size(); ++i) {
    if (fileName == toc[i].name) {
      file = &toc[i];
      break;
    }
  }
  if (!file) {
    return mlir::emitError(
        loc, "unable to merge builtin module; file not found " + fileName);
  }
  return mergeSourceModuleInto(loc, StringRef(file->data, file->size), targetOp,
                               targetBuilder);
}

//===----------------------------------------------------------------------===//
// stream.builtin.splat.i64
//===----------------------------------------------------------------------===//

LogicalResult BuiltinSplatI64Op::verify() {
  BuiltinSplatI64Op op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

LogicalResult BuiltinSplatI64Op::mergeBuiltinModule(Operation *targetOp,
                                                    OpBuilder &targetBuilder) {
  return mergeBuiltinModuleSource(getLoc(), "splat_i64.mlir", targetOp,
                                  targetBuilder);
}

LogicalResult BuiltinSplatI64Op::convertBuiltinOp(OpBuilder &builder) {
  auto c8 = builder.createOrFold<arith::ConstantIndexOp>(getLoc(), 8);
  auto count =
      builder.createOrFold<arith::DivUIOp>(getLoc(), getResultSize(), c8);
  Value workload[] = {count};
  SmallVector<Value> operands = {
      getValue(),
      count,
  };
  SmallVector<Value> operandSizes;
  SmallVector<Value> operandOffsets;
  SmallVector<Value> operandEnds;
  SmallVector<Value> operandLengths;
  SmallVector<int64_t> tiedOperands = {
      -1,
  };
  SmallVector<Value> resultSizes = {
      getResultSize(),
  };
  SmallVector<Type> resultTypes{
      getResult().getType(),
  };
  auto dispatchOp = builder.create<IREE::Stream::AsyncDispatchOp>(
      getLoc(), resultTypes, workload,
      SymbolRefAttr::get(
          builder.getStringAttr("__builtin_splat_i64"),
          FlatSymbolRefAttr::get(builder.getContext(), "__builtin_splat_i64")),
      operands, operandSizes, operandOffsets, operandEnds, operandLengths,
      resultSizes, builder.getIndexArrayAttr(tiedOperands), getAffinityAttr());
  getResult().replaceAllUsesWith(dispatchOp.getResults().front());
  return success();
}

//===----------------------------------------------------------------------===//
// stream.builtin.fill.i64
//===----------------------------------------------------------------------===//

LogicalResult BuiltinFillI64Op::verify() {
  BuiltinFillI64Op op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value BuiltinFillI64Op::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> BuiltinFillI64Op::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> BuiltinFillI64Op::getTiedResultOperandIndices() {
  return {0};  // target
}

LogicalResult BuiltinFillI64Op::mergeBuiltinModule(Operation *targetOp,
                                                   OpBuilder &targetBuilder) {
  return mergeBuiltinModuleSource(getLoc(), "fill_i64.mlir", targetOp,
                                  targetBuilder);
}

LogicalResult BuiltinFillI64Op::convertBuiltinOp(OpBuilder &builder) {
  auto c8 = builder.createOrFold<arith::ConstantIndexOp>(getLoc(), 8);
  auto count =
      builder.createOrFold<arith::DivUIOp>(getLoc(), getTargetLength(), c8);
  Value workload[] = {count};
  SmallVector<Value> operands = {
      getTarget(),
      getValue(),
      getTargetOffset(),
      count,
  };
  SmallVector<Value> operandSizes = {
      getTargetSize(),
  };
  SmallVector<Value> operandOffsets = {
      getTargetOffset(),
  };
  SmallVector<Value> operandEnds = {
      getTargetEnd(),
  };
  SmallVector<Value> operandLengths = {
      getTargetLength(),
  };
  SmallVector<int64_t> tiedOperands = {
      0,
  };
  SmallVector<Value> resultSizes = {
      getTargetSize(),
  };
  SmallVector<Type> resultTypes{
      getResult().getType(),
  };
  auto dispatchOp = builder.create<IREE::Stream::AsyncDispatchOp>(
      getLoc(), resultTypes, workload,
      SymbolRefAttr::get(
          builder.getStringAttr("__builtin_fill_i64"),
          FlatSymbolRefAttr::get(builder.getContext(), "__builtin_fill_i64")),
      operands, operandSizes, operandOffsets, operandEnds, operandLengths,
      resultSizes, builder.getIndexArrayAttr(tiedOperands), getAffinityAttr());
  getResult().replaceAllUsesWith(dispatchOp.getResults().front());
  return success();
}

//===----------------------------------------------------------------------===//
// stream.async.alloca
//===----------------------------------------------------------------------===//

bool AsyncAllocaOp::isMetadata() { return true; }

bool AsyncAllocaOp::preferCloneToConsumers() { return true; }

//===----------------------------------------------------------------------===//
// stream.async.constant
//===----------------------------------------------------------------------===//

bool AsyncConstantOp::isMetadata() { return true; }

void AsyncConstantOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "cst");
}

LogicalResult AsyncConstantOp::verify() {
  AsyncConstantOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

void AsyncConstantOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Write, getResult(), Value{},
                    getResultSize(), getResultSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.splat
//===----------------------------------------------------------------------===//

LogicalResult AsyncSplatOp::verify() {
  AsyncSplatOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

bool AsyncSplatOp::preferCloneToConsumers() { return true; }

void AsyncSplatOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Write, getResult(), Value{},
                    getResultSize(), getResultSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.clone
//===----------------------------------------------------------------------===//

LogicalResult AsyncCloneOp::verify() {
  AsyncCloneOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

bool AsyncCloneOp::preferCloneToConsumers() { return true; }

void AsyncCloneOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getSource(), Value{},
                    getSourceSize(), getSourceSize()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(), Value{},
                    getResultSize(), getResultSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.slice
//===----------------------------------------------------------------------===//

LogicalResult AsyncSliceOp::verify() {
  AsyncSliceOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

void AsyncSliceOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getSource(),
                    getSourceOffset(), getSourceEnd(), getResultSize()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(), Value{},
                    getResultSize(), getResultSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.fill
//===----------------------------------------------------------------------===//

LogicalResult AsyncFillOp::verify() {
  AsyncFillOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value AsyncFillOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> AsyncFillOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncFillOp::getTiedResultOperandIndices() {
  return {0};  // target
}

void AsyncFillOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Write, getTarget(),
                    getTargetOffset(), getTargetEnd(), getTargetLength()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(),
                    getTargetOffset(), getTargetEnd(), getTargetLength()});
}

//===----------------------------------------------------------------------===//
// stream.async.update
//===----------------------------------------------------------------------===//

LogicalResult AsyncUpdateOp::verify() {
  AsyncUpdateOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getUpdate(), op.getUpdateSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value AsyncUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> AsyncUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

void AsyncUpdateOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getUpdate(), Value{},
                    getUpdateSize(), getUpdateSize()});
  ranges.push_back({ResourceAccessBitfield::Write, getTarget(),
                    getTargetOffset(), getTargetEnd(), getUpdateSize()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(),
                    getTargetOffset(), getTargetEnd(), getUpdateSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.copy
//===----------------------------------------------------------------------===//

LogicalResult AsyncCopyOp::verify() {
  AsyncCopyOp op = *this;
  if (op.getSource() == op.getTarget()) {
    // If we want to perform memmove-like operations where it's safe to copy
    // overlapping ranges we'll need to emit some runtime checks. We can in
    // many cases statically detect a lack of overlap just based on symbolic
    // offset equality but that requires some analysis we don't have yet.
    return op.emitOpError() << "cannot copy within the same resource (yet)";
  }
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value AsyncCopyOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> AsyncCopyOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncCopyOp::getTiedResultOperandIndices() {
  return {0};  // target
}

void AsyncCopyOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getSource(),
                    getSourceOffset(), getSourceEnd(), getLength()});
  ranges.push_back({ResourceAccessBitfield::Write, getTarget(),
                    getTargetOffset(), getTargetEnd(), getLength()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(),
                    getTargetOffset(), getTargetEnd(), getLength()});
}

//===----------------------------------------------------------------------===//
// stream.async.collective
//===----------------------------------------------------------------------===//

static const char *getCollectiveParamKeyword(Attribute opAttr) {
  auto attr = opAttr.cast<IREE::Stream::CollectiveAttr>();
  switch (attr.getKind()) {
    case IREE::Stream::CollectiveKind::Broadcast:
      return "source";
    case IREE::Stream::CollectiveKind::Reduce:
      return "target";
    case IREE::Stream::CollectiveKind::Send:
      return "target";
    case IREE::Stream::CollectiveKind::Recv:
      return "source";
    default:
      return nullptr;
  }
}

static ParseResult parseCollectiveParam(
    OpAsmParser &parser, Attribute opAttr,
    Optional<OpAsmParser::UnresolvedOperand> &optionalParamValue) {
  const char *keyword = getCollectiveParamKeyword(opAttr);
  if (!keyword) return success();  // optional
  OpAsmParser::UnresolvedOperand paramValue;
  if (failed(parser.parseKeyword(keyword)) || failed(parser.parseLParen()) ||
      failed(parser.parseOperand(paramValue)) || failed(parser.parseRParen())) {
    return failure();
  }
  optionalParamValue = paramValue;
  return success();
}

static void printCollectiveParam(OpAsmPrinter &p, Operation *op,
                                 Attribute opAttr, Value paramValue) {
  const char *keyword = getCollectiveParamKeyword(opAttr);
  if (paramValue) {
    assert(keyword && "collective op must have a param keyword");
    p << keyword << "(";
    p.printOperand(paramValue);
    p << ")";
  }
}

LogicalResult AsyncCollectiveOp::verify() {
  AsyncCollectiveOp op = *this;

  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getTargetSize()))) {
    return failure();
  }

  const bool hasParam = !!op.getParam();
  const char *paramKeyword = getCollectiveParamKeyword(op.getOp());
  if (paramKeyword) {
    // Param required.
    if (!hasParam) {
      return op.emitOpError() << "collective operation requires a "
                              << paramKeyword << " parameter but none present";
    }
  } else {
    // No param required.
    if (hasParam) {
      return op.emitOpError() << "collective operation does not require a "
                                 "parameter but one present";
    }
  }

  return success();
}

Value AsyncCollectiveOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> AsyncCollectiveOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncCollectiveOp::getTiedResultOperandIndices() {
  return {0};  // target
}

void AsyncCollectiveOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getSource(),
                    getSourceOffset(), getSourceEnd(), getSourceLength()});
  ranges.push_back({ResourceAccessBitfield::Write, getTarget(),
                    getTargetOffset(), getTargetEnd(), getTargetLength()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(),
                    getTargetOffset(), getTargetEnd(), getTargetLength()});
}

//===----------------------------------------------------------------------===//
// stream.async.transfer
//===----------------------------------------------------------------------===//

LogicalResult AsyncTransferOp::verify() {
  AsyncTransferOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getResult(), op.getResultSize()))) {
    return failure();
  }
  return success();
}

void AsyncTransferOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  ranges.push_back({ResourceAccessBitfield::Read, getSource(), Value{},
                    getSourceSize(), getSourceSize()});
  ranges.push_back({ResourceAccessBitfield::Write, getResult(), Value{},
                    getResultSize(), getResultSize()});
}

//===----------------------------------------------------------------------===//
// stream.async.load
//===----------------------------------------------------------------------===//

LogicalResult AsyncLoadOp::verify() {
  AsyncLoadOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.async.store
//===----------------------------------------------------------------------===//

LogicalResult AsyncStoreOp::verify() {
  AsyncStoreOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

Value AsyncStoreOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

::llvm::Optional<unsigned> AsyncStoreOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncStoreOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.async.dispatch
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceOperands,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceEnds,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceLengths) {
  do {
    // All entries at least have an %operand.
    resourceOperands.emplace_back();
    if (failed(parser.parseOperand(resourceOperands.back()))) return failure();
    // Resources have a range.
    if (succeeded(parser.parseOptionalLSquare())) {
      resourceOffsets.emplace_back();
      resourceEnds.emplace_back();
      resourceLengths.emplace_back();
      if (failed(parser.parseOperand(resourceOffsets.back())) ||
          failed(parser.parseKeyword("to")) ||
          failed(parser.parseOperand(resourceEnds.back())) ||
          failed(parser.parseKeyword("for")) ||
          failed(parser.parseOperand(resourceLengths.back())) ||
          failed(parser.parseRSquare())) {
        return failure();
      }
    }
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printDispatchOperands(OpAsmPrinter &p, Operation *op,
                                  ValueRange resourceOperands,
                                  ValueRange resourceOffsets,
                                  ValueRange resourceEnds,
                                  ValueRange resourceLengths) {
  unsigned resourceIndex = 0;
  llvm::interleaveComma(resourceOperands, p, [&](Value operand) {
    p.printOperand(operand);
    if (operand.getType().isa<IREE::Stream::ResourceType>()) {
      p << "[";
      p.printOperand(resourceOffsets[resourceIndex]);
      p << " to ";
      p.printOperand(resourceEnds[resourceIndex]);
      p << " for ";
      p.printOperand(resourceLengths[resourceIndex]);
      p << "]";
      ++resourceIndex;
    }
  });
}

LogicalResult AsyncDispatchOp::verify() {
  AsyncDispatchOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResourceOperands(),
                                op.getResourceOperandSizes())) ||
      failed(verifyOpValueSizes(op, op.getResults(), op.getResultSizes()))) {
    return failure();
  }
  unsigned requiredRangeCount = 0;
  for (auto value : op.getResourceOperands()) {
    if (value.getType().isa<IREE::Stream::ResourceType>()) {
      ++requiredRangeCount;
    }
  }
  unsigned presentRangeCount = op.getResourceOperandOffsets().size();
  if (op.getResourceOperandEnds().size() != presentRangeCount ||
      op.getResourceOperandLengths().size() != presentRangeCount) {
    return op->emitOpError() << "mismatch on resource range "
                                "offsets/ends/lengths; counts must match";
  }
  if (presentRangeCount != requiredRangeCount) {
    return op->emitOpError() << "expects " << requiredRangeCount
                             << " resource range operand sets but "
                             << presentRangeCount << " are present";
  }
  return success();
}

LogicalResult AsyncDispatchOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto exportOp =
      symbolTable.lookupNearestSymbolFrom<IREE::Stream::ExecutableExportOp>(
          op, getEntryPoint());
  if (!exportOp) {
    // TODO(benvanik): there are a lot of tests that are assuming this is not
    // verified. We'll need to go add dummy executables for all of them. Today
    // we just bail on the verifier if the symbol isn't found.
    //
    // Should be:
    //   return op->emitOpError() << "undefined entry point: " << entry_point();
    return success();
  }

  // Verify that the workload parameters captured match the target export.
  if (failed(verifyDispatchWorkload(op, exportOp, getWorkload()))) {
    return failure();
  }

  // TODO(benvanik): verify that the target function has matching operands.

  return success();
}

std::pair<unsigned, unsigned> AsyncDispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);  // $operands
}

void AsyncDispatchOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  unsigned rangeIndex = 0;
  unsigned tiedOperandBase = getTiedOperandsIndexAndLength().first;
  for (auto [operandIndex, operand] : llvm::enumerate(getResourceOperands())) {
    if (!operand.getType().isa<IREE::Stream::ResourceType>()) continue;
    ResourceAccessBitfield access = ResourceAccessBitfield::Read;
    auto tiedResults = getOperandTiedResults(tiedOperandBase + operandIndex);
    if (!tiedResults.empty()) {
      access = access | ResourceAccessBitfield::Write;
    }
    Value start = getResourceOperandOffsets()[rangeIndex];
    Value end = getResourceOperandEnds()[rangeIndex];
    Value length = getResourceOperandLengths()[rangeIndex];
    ++rangeIndex;
    ranges.push_back({access, operand, start, end, length});
    for (auto result : tiedResults) {
      ranges.push_back({access, result, start, end, length});
    }
  }
  for (auto [i, result, resultSize] :
       llvm::zip_equal(llvm::seq<unsigned>(0, getResults().size()),
                       getResults(), getResultSizes())) {
    if (getTiedResultOperandIndex(i).has_value()) {
      // Already covered above.
      continue;
    }
    ranges.push_back({ResourceAccessBitfield::Write, result, Value{},
                      resultSize, resultSize});
  }
}

//===----------------------------------------------------------------------===//
// stream.async.execute
//===----------------------------------------------------------------------===//

void AsyncExecuteOp::build(OpBuilder &builder, OperationState &state,
                           TypeRange resultTypes, ValueRange resultSizes,
                           Value awaitTimepoint, ValueRange operands,
                           ValueRange operandSizes,
                           ArrayRef<int64_t> tiedOperands,
                           ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addTypes(IREE::Stream::TimepointType::get(builder.getContext()));
  state.addOperands(operands);
  state.addOperands(operandSizes);
  state.addOperands(resultSizes);
  if (awaitTimepoint) state.addOperands(awaitTimepoint);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(resultSizes.size()),
                         awaitTimepoint ? 1 : 0,
                     }));
  state.addRegion();
}

LogicalResult AsyncExecuteOp::verify() {
  AsyncExecuteOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResourceOperands(),
                                op.getResourceOperandSizes())) ||
      failed(verifyOpValueSizes(op, op.getResults(), op.getResultSizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.getBody())) ||
      failed(verifyEscapingResources(op.getBody(), op.getResults(),
                                     op.getResultSizes()))) {
    return failure();
  }
  return success();
}

std::pair<unsigned, unsigned> AsyncExecuteOp::getTiedResultsIndexAndLength() {
  return {0, getResults().size()};
}

OperandRange AsyncExecuteOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  assert(index && index.value() == 0 && "invalid region index");
  return getResourceOperands();
}

void AsyncExecuteOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.has_value()) {
    regions.push_back(RegionSuccessor(getResults()));
  } else {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
  }
}

// Gets the async access ranges for the generic stream execution op capturing
// resources.
template <typename Op>
static void getExecutionAsyncAccessRanges(
    Op op, SmallVectorImpl<AsyncAccessRange> &ranges) {
  unsigned tiedOperandBase = op.getTiedOperandsIndexAndLength().first;
  for (auto [i, operand, operandSize] : llvm::zip_equal(
           llvm::seq<unsigned>(0, op.getResourceOperands().size()),
           op.getResourceOperands(), op.getResourceOperandSizes())) {
    if (!operand.getType().template isa<IREE::Stream::ResourceType>()) continue;
    ResourceAccessBitfield access = ResourceAccessBitfield::Read;
    auto tiedResults = op.getOperandTiedResults(tiedOperandBase + i);
    if (!tiedResults.empty()) {
      access = access | ResourceAccessBitfield::Write;
    }
    ranges.push_back({access, operand, Value{}, operandSize, operandSize});
    for (auto result : tiedResults) {
      ranges.push_back({access, result, Value{}, operandSize, operandSize});
    }
  }
  for (auto [i, result, resultSize] :
       llvm::zip_equal(llvm::seq<unsigned>(0, op.getResults().size()),
                       op.getResults(), op.getResultSizes())) {
    if (op.getTiedResultOperandIndex(i).has_value()) {
      // Already covered above.
      continue;
    }
    ranges.push_back({ResourceAccessBitfield::Write, result, Value{},
                      resultSize, resultSize});
  }
}

void AsyncExecuteOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  getExecutionAsyncAccessRanges(*this, ranges);
}

Operation::operand_range AsyncExecuteOp::getClosureOperands() {
  return getResourceOperands();
}

Operation::result_range AsyncExecuteOp::getClosureResults() {
  return getResults();
}

bool AsyncExecuteOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess AsyncExecuteOp::getOperandAccess(
    unsigned operandIndex) {
  auto arg = getBody().getArgument(operandIndex);
  return computeValueAccess(arg);
}

IREE::Util::ValueAccess AsyncExecuteOp::getResultAccess(unsigned resultIndex) {
  auto yieldOp = cast<YieldOp>(getBody().getBlocks().front().getTerminator());
  return computeValueAccess(yieldOp.getOperand(resultIndex));
}

IREE::Util::ClosureOpInterface
AsyncExecuteOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  auto newResultTypes = llvm::to_vector<4>(llvm::map_range(
      getResults(), [](auto value) { return value.getType(); }));
  auto newResultSizes = llvm::to_vector<4>(getResultSizes());
  auto newOperandsValues = llvm::to_vector<4>(getResourceOperands());
  auto newOperandSizes = llvm::to_vector<4>(getResourceOperandSizes());
  IREE::Util::excludeClosureOperandsAndResults(
      newOperandsValues, newOperandSizes, excludedOperandIndices,
      newResultTypes, newResultSizes, excludedResultIndices);

  auto newTiedOperandIndices =
      llvm::to_vector<4>(getTiedResultOperandIndices());
  IREE::Util::excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);
  assert(getTiedOperandsIndexAndLength().first == 0 &&
         "operands must be the first ODS group");

  auto newOp = rewriter.create<AsyncExecuteOp>(
      getLoc(), newResultTypes, newResultSizes, getAwaitTimepoint(),
      newOperandsValues, newOperandSizes, newTiedOperandIndices,
      getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  eraseStreamRegionResults(newBody, excludedResultIndices);

  auto &block = newBody.front();
  BitVector eraseIndices(block.getNumArguments());
  for (auto i : excludedOperandIndices) eraseIndices.set(i);
  block.eraseArguments(eraseIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// stream.async.concurrent
//===----------------------------------------------------------------------===//

void AsyncConcurrentOp::build(OpBuilder &builder, OperationState &state,
                              TypeRange resultTypes, ValueRange resultSizes,
                              ValueRange operands, ValueRange operandSizes,
                              ArrayRef<int64_t> tiedOperands,
                              ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandSizes);
  state.addOperands(resultSizes);
  state.addAttributes(attributes);
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(resultSizes.size()),
                     }));
  state.addRegion();
}

LogicalResult AsyncConcurrentOp::verify() {
  AsyncConcurrentOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResourceOperands(),
                                op.getResourceOperandSizes())) ||
      failed(verifyOpValueSizes(op, op.getResults(), op.getResultSizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.getBody())) ||
      failed(verifyEscapingResources(op.getBody(), op.getResults(),
                                     op.getResultSizes()))) {
    return failure();
  }
  return success();
}

OperandRange AsyncConcurrentOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  assert(index && index.value() == 0 && "invalid region index");
  return getResourceOperands();
}

void AsyncConcurrentOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.has_value()) {
    regions.push_back(RegionSuccessor(getResults()));
  } else {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
  }
}

void AsyncConcurrentOp::getAsyncAccessRanges(
    SmallVectorImpl<AsyncAccessRange> &ranges) {
  getExecutionAsyncAccessRanges(*this, ranges);
}

Operation::operand_range AsyncConcurrentOp::getClosureOperands() {
  return getResourceOperands();
}

Operation::result_range AsyncConcurrentOp::getClosureResults() {
  return getResults();
}

bool AsyncConcurrentOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess AsyncConcurrentOp::getOperandAccess(
    unsigned operandIndex) {
  auto arg = getBody().getArgument(operandIndex);
  return computeValueAccess(arg);
}

IREE::Util::ValueAccess AsyncConcurrentOp::getResultAccess(
    unsigned resultIndex) {
  auto yieldOp = cast<YieldOp>(getBody().getBlocks().front().getTerminator());
  return computeValueAccess(yieldOp.getOperand(resultIndex));
}

IREE::Util::ClosureOpInterface
AsyncConcurrentOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  auto newResultTypes = llvm::to_vector<4>(getResultTypes());
  auto newResultSizes = llvm::to_vector<4>(getResultSizes());
  auto newOperandsValues = llvm::to_vector<4>(getResourceOperands());
  auto newOperandSizes = llvm::to_vector<4>(getResourceOperandSizes());
  IREE::Util::excludeClosureOperandsAndResults(
      newOperandsValues, newOperandSizes, excludedOperandIndices,
      newResultTypes, newResultSizes, excludedResultIndices);

  auto newTiedOperandIndices =
      llvm::to_vector<4>(getTiedResultOperandIndices());
  IREE::Util::excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);
  assert(getTiedOperandsIndexAndLength().first == 0 &&
         "operands must be the first ODS group");

  auto newOp = rewriter.create<AsyncConcurrentOp>(
      getLoc(), newResultTypes, newResultSizes, newOperandsValues,
      newOperandSizes, newTiedOperandIndices, getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  eraseStreamRegionResults(newBody, excludedResultIndices);
  auto &block = newBody.front();
  BitVector eraseIndices(block.getNumArguments());
  for (auto i : excludedOperandIndices) eraseIndices.set(i);
  block.eraseArguments(eraseIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// stream.cmd.flush
//===----------------------------------------------------------------------===//

LogicalResult CmdFlushOp::verify() {
  CmdFlushOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.invalidate
//===----------------------------------------------------------------------===//

LogicalResult CmdInvalidateOp::verify() {
  CmdInvalidateOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.discard
//===----------------------------------------------------------------------===//

LogicalResult CmdDiscardOp::verify() {
  CmdDiscardOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.fill
//===----------------------------------------------------------------------===//

LogicalResult CmdFillOp::verify() {
  CmdFillOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.copy
//===----------------------------------------------------------------------===//

LogicalResult CmdCopyOp::verify() {
  CmdCopyOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getSource(), op.getSourceSize())) ||
      failed(verifyOpValueSizes(op, op.getTarget(), op.getTargetSize()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.collective
//===----------------------------------------------------------------------===//

LogicalResult CmdCollectiveOp::verify() {
  CmdCollectiveOp op = *this;
  size_t resourceCount = op.getResources().size();
  if (op.getResourceSizes().size() != resourceCount ||
      op.getResourceOffsets().size() != resourceCount ||
      op.getResourceLengths().size() != resourceCount ||
      op.getResourceAccesses().size() != resourceCount) {
    return op->emitOpError() << "with " << resourceCount
                             << " resources has mismatched associated ranges";
  }

  size_t requiredCount = 0;
  IREE::Stream::ResourceAccessBitfield requiredAccess[2] = {
      IREE::Stream::ResourceAccessBitfield::None,
      IREE::Stream::ResourceAccessBitfield::None,
  };
  switch (getOp().getKind()) {
    default:
      requiredCount = 2;  // send & recv
      requiredAccess[0] = IREE::Stream::ResourceAccessBitfield::Read;
      requiredAccess[1] = IREE::Stream::ResourceAccessBitfield::Write;
      break;
    case IREE::Stream::CollectiveKind::Send:
      requiredCount = 1;  // send
      requiredAccess[0] = IREE::Stream::ResourceAccessBitfield::Read;
      break;
    case IREE::Stream::CollectiveKind::Recv:
      requiredCount = 1;  // recv
      requiredAccess[0] = IREE::Stream::ResourceAccessBitfield::Write;
      break;
  }
  if (resourceCount != requiredCount) {
    return op->emitOpError()
           << "requires " << requiredCount << " resources but " << resourceCount
           << " provided";
  }
  for (size_t i = 0; i < requiredCount; ++i) {
    auto declaredAccess = op.getResourceAccesses()[i]
                              .cast<IREE::Stream::ResourceAccessBitfieldAttr>()
                              .getValue();
    if (!bitEnumContainsAll(declaredAccess, requiredAccess[i])) {
      return op->emitOpError()
             << "resource " << i << " requires access "
             << stringifyEnum(requiredAccess[i]) << " but is declared as "
             << stringifyEnum(declaredAccess);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.dispatch
//===----------------------------------------------------------------------===//

LogicalResult CmdDispatchOp::verify() {
  CmdDispatchOp op = *this;
  size_t resourceCount = op.getResources().size();
  if (op.getResourceSizes().size() != resourceCount ||
      op.getResourceOffsets().size() != resourceCount ||
      op.getResourceLengths().size() != resourceCount ||
      op.getResourceAccesses().size() != resourceCount) {
    return op->emitOpError() << "dispatch with " << resourceCount
                             << " resources has mismatched associated ranges";
  }
  return success();
}

LogicalResult CmdDispatchOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto entryPointRefs = getEntryPointRefs();
  if (entryPointRefs.empty()) {
    return emitOpError() << "at least one entry point must be defined";
  }
  for (auto entryPointAttr : entryPointRefs) {
    auto exportOp =
        symbolTable.lookupNearestSymbolFrom<IREE::Stream::ExecutableExportOp>(
            op, entryPointAttr);
    if (!exportOp) {
      // TODO(benvanik): there are a lot of tests that are assuming this is not
      // verified. We'll need to go add dummy executables for all of them. Today
      // we just bail on the verifier if the symbol isn't found.
      //
      // Should be:
      //   return op->emitOpError() << "undefined entry point: " <<
      //   entry_point();
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

static ParseResult parseDispatchEntryPoints(OpAsmParser &parser,
                                            ArrayAttr &entryPointAttrsArray) {
  SmallVector<Attribute> entryPointAttrs;
  if (succeeded(parser.parseOptionalLBrace())) {
    do {
      SymbolRefAttr entryPointAttr;
      if (failed(parser.parseAttribute(entryPointAttr))) return failure();
      entryPointAttrs.push_back(entryPointAttr);
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRBrace())) return failure();
  } else {
    SymbolRefAttr entryPointAttr;
    if (failed(parser.parseAttribute(entryPointAttr))) return failure();
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

static ParseResult parseDispatchResources(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resources,
    SmallVectorImpl<Type> &resourceTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceSizes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resourceLengths,
    ArrayAttr &resourceAccesses) {
  SmallVector<Attribute> accessAttrs;
  do {
    // Reserve entries in the lists.
    resources.emplace_back();
    resourceTypes.emplace_back();
    resourceSizes.emplace_back();
    resourceOffsets.emplace_back();
    resourceLengths.emplace_back();
    StringRef accessStr;
    if (failed(parser.parseKeyword(&accessStr)) ||
        failed(parser.parseOperand(resources.back())) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(resourceOffsets.back())) ||
        failed(parser.parseKeyword("for")) ||
        failed(parser.parseOperand(resourceLengths.back())) ||
        failed(parser.parseRSquare()) || failed(parser.parseColon()) ||
        failed(parseSizeAwareType(parser, resourceTypes.back(),
                                  resourceSizes.back()))) {
      return failure();
    }
    IREE::Stream::ResourceAccessBitfield accessBits =
        IREE::Stream::ResourceAccessBitfield::None;
    if (accessStr == "ro") {
      accessBits = IREE::Stream::ResourceAccessBitfield::Read;
    } else if (accessStr == "wo") {
      accessBits = IREE::Stream::ResourceAccessBitfield::Write;
    } else if (accessStr == "rw") {
      accessBits = IREE::Stream::ResourceAccessBitfield::Read |
                   IREE::Stream::ResourceAccessBitfield::Write;
    }
    accessAttrs.push_back(IREE::Stream::ResourceAccessBitfieldAttr::get(
        parser.getBuilder().getContext(), accessBits));
  } while (succeeded(parser.parseOptionalComma()));
  resourceAccesses = parser.getBuilder().getArrayAttr(accessAttrs);
  return success();
}

static void printDispatchResources(OpAsmPrinter &p, Operation *op,
                                   ValueRange resources,
                                   TypeRange resourceTypes,
                                   ValueRange resourceSizes,
                                   ValueRange resourceOffsets,
                                   ValueRange resourceLengths,
                                   ArrayAttr resourceAccesses) {
  for (size_t i = 0; i < resources.size(); ++i) {
    auto resource = resources[i];
    auto resourceType = resourceTypes[i];
    auto resourceSize = resourceSizes[i];
    auto resourceOffset = resourceOffsets[i];
    auto resourceLength = resourceLengths[i];
    auto resourceAccess = resourceAccesses[i]
                              .cast<IREE::Stream::ResourceAccessBitfieldAttr>()
                              .getValue();
    p.printNewline();
    p << "  ";
    if (bitEnumContainsAll(resourceAccess,
                           IREE::Stream::ResourceAccessBitfield::Read |
                               IREE::Stream::ResourceAccessBitfield::Write)) {
      p << "rw";
    } else if (bitEnumContainsAll(resourceAccess,
                                  IREE::Stream::ResourceAccessBitfield::Read)) {
      p << "ro";
    } else if (bitEnumContainsAll(
                   resourceAccess,
                   IREE::Stream::ResourceAccessBitfield::Write)) {
      p << "wo";
    }
    p << ' ';
    p.printOperand(resource);
    p << "[";
    p.printOperand(resourceOffset);
    p << " for ";
    p.printOperand(resourceLength);
    p << "] : ";
    printSizeAwareType(p, op, resourceType, resourceSize);
    if (i < resources.size() - 1) p << ",";
  }
}

// This is sloppy because the function has interleaved bindings and operands;
// if we had our own op we could just reuse the map we have for operands.
// static
SmallVector<unsigned> CmdDispatchOp::makeOperandToArgMap(
    mlir::func::FuncOp funcOp) {
  unsigned operandCount = llvm::count_if(
      funcOp.getArgumentTypes(),
      [](Type type) { return !type.isa<IREE::Stream::BindingType>(); });
  SmallVector<unsigned> map(operandCount);
  unsigned operandIdx = 0;
  for (auto it : llvm::enumerate(funcOp.getArgumentTypes())) {
    unsigned argIdx = it.index();
    auto argType = it.value();
    if (!argType.isa<IREE::Stream::BindingType>()) {
      map[operandIdx++] = argIdx;
    }
  }
  return map;
}

// static
SmallVector<unsigned> CmdDispatchOp::makeResourceToArgMap(
    mlir::func::FuncOp funcOp) {
  unsigned operandCount = llvm::count_if(
      funcOp.getArgumentTypes(),
      [](Type type) { return type.isa<IREE::Stream::BindingType>(); });
  SmallVector<unsigned> map(operandCount);
  unsigned operandIdx = 0;
  for (auto it : llvm::enumerate(funcOp.getArgumentTypes())) {
    unsigned argIdx = it.index();
    auto argType = it.value();
    if (argType.isa<IREE::Stream::BindingType>()) {
      map[operandIdx++] = argIdx;
    }
  }
  return map;
}

//===----------------------------------------------------------------------===//
// stream.cmd.execute
//===----------------------------------------------------------------------===//

void CmdExecuteOp::build(OpBuilder &builder, OperationState &state,
                         Value awaitTimepoint, ValueRange operands,
                         ValueRange operandSizes,
                         ArrayRef<NamedAttribute> attributes) {
  state.addTypes(IREE::Stream::TimepointType::get(builder.getContext()));
  state.addOperands(operands);
  state.addOperands(operandSizes);
  if (awaitTimepoint) state.addOperands(awaitTimepoint);
  state.addAttributes(attributes);
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         awaitTimepoint ? 1 : 0,
                     }));
  state.addRegion();
}

// Returns success if the given op is a known valid stream.cmd.* op for use
// within an execution region.
static LogicalResult verifyCmdOp(Operation *op) {
  if (!op->hasTrait<OpTrait::IREE::Stream::CmdPhaseOp>() &&
      !isa<IREE::Stream::StreamableOpInterface>(op) &&
      !isa<IREE::Stream::YieldOp>(op)) {
    return op->emitOpError()
           << "explicit execution regions must only contain explicit ops";
  }
  return success();
}

LogicalResult CmdExecuteOp::verify() {
  CmdExecuteOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResourceOperands(),
                                op.getResourceOperandSizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.getBody()))) {
    return failure();
  }
  for (auto &nestedOp : op.getBody().front()) {
    if (failed(verifyCmdOp(&nestedOp))) return failure();
  }
  return success();
}

OperandRange CmdExecuteOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && index.value() == 0 && "invalid region index");
  return getResourceOperands();
}

void CmdExecuteOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.has_value()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
  }
}

Operation::operand_range CmdExecuteOp::getClosureOperands() {
  return getResourceOperands();
}

Operation::result_range CmdExecuteOp::getClosureResults() {
  return Operation::result_range(nullptr, 0);
}

bool CmdExecuteOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess CmdExecuteOp::getOperandAccess(unsigned operandIndex) {
  auto arg = getBody().getArgument(operandIndex);
  return computeValueAccess(arg);
}

IREE::Util::ValueAccess CmdExecuteOp::getResultAccess(unsigned resultIndex) {
  return IREE::Util::ValueAccess::None();
}

IREE::Util::ClosureOpInterface
CmdExecuteOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  SmallVector<Type, 4> newResultTypes;
  SmallVector<Value, 4> newResultSizes;
  auto newOperandsValues = llvm::to_vector<4>(getResourceOperands());
  auto newOperandSizes = llvm::to_vector<4>(getResourceOperandSizes());
  IREE::Util::excludeClosureOperandsAndResults(
      newOperandsValues, newOperandSizes, excludedOperandIndices,
      newResultTypes, newResultSizes, excludedResultIndices);

  auto newOp = rewriter.create<CmdExecuteOp>(getLoc(), getAwaitTimepoint(),
                                             newOperandsValues, newOperandSizes,
                                             getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  auto &block = newBody.front();
  BitVector eraseIndices(block.getNumArguments());
  for (auto i : excludedOperandIndices) eraseIndices.set(i);
  block.eraseArguments(eraseIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// stream.cmd.serial
//===----------------------------------------------------------------------===//

LogicalResult CmdSerialOp::verify() {
  CmdSerialOp op = *this;
  for (auto &nestedOp : op.getBody().front()) {
    if (failed(verifyCmdOp(&nestedOp))) return failure();
  }
  return success();
}

void CmdSerialOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.has_value()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&getBody(), {}));
  }
}

//===----------------------------------------------------------------------===//
// stream.cmd.concurrent
//===----------------------------------------------------------------------===//

LogicalResult CmdConcurrentOp::verify() {
  CmdConcurrentOp op = *this;
  for (auto &nestedOp : op.getBody().front()) {
    if (failed(verifyCmdOp(&nestedOp))) return failure();
  }
  return success();
}

void CmdConcurrentOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.has_value()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&getBody(), {}));
  }
}

//===----------------------------------------------------------------------===//
// stream.timepoint.join
//===----------------------------------------------------------------------===//

LogicalResult TimepointJoinOp::verify() {
  // We could test if timepoints all come from the same place - this is not
  // strictly required but if we could avoid it things will be easier to
  // implement at runtime (won't have to do a cuda<->vulkan sync, etc).
  return success();
}

//===----------------------------------------------------------------------===//
// stream.timepoint.barrier
//===----------------------------------------------------------------------===//

LogicalResult TimepointBarrierOp::verify() {
  TimepointBarrierOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResource(), op.getResourceSize()))) {
    return failure();
  }
  return success();
}

Value TimepointBarrierOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getResource());
}

::llvm::Optional<unsigned> TimepointBarrierOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};
}

SmallVector<int64_t, 4> TimepointBarrierOp::getTiedResultOperandIndices() {
  return {0};
}

std::pair<unsigned, unsigned>
TimepointBarrierOp::getTiedResultsIndexAndLength() {
  return {0, 1};
}

//===----------------------------------------------------------------------===//
// stream.timepoint.await
//===----------------------------------------------------------------------===//

void TimepointAwaitOp::build(OpBuilder &builder, OperationState &state,
                             ValueRange operands, ValueRange operandSizes,
                             Value timepoint,
                             ArrayRef<NamedAttribute> attributes) {
  state.addTypes(llvm::map_range(
      operands, [&](Value operand) { return operand.getType(); }));
  state.addOperands(operands);
  state.addOperands(operandSizes);
  state.addOperands(timepoint);
  state.addAttributes(attributes);
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(1),  // timepoint
                     }));
}

LogicalResult TimepointAwaitOp::verify() {
  TimepointAwaitOp op = *this;
  if (failed(verifyOpValueSizes(op, op.getResourceOperands(),
                                op.getResourceOperandSizes())) ||
      failed(verifyOpValueSizes(op, op.getResults(),
                                op.getResourceOperandSizes()))) {
    return failure();
  }
  return success();
}

::llvm::Optional<unsigned> TimepointAwaitOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {resultIndex};
}

SmallVector<int64_t, 4> TimepointAwaitOp::getTiedResultOperandIndices() {
  return llvm::to_vector<4>(
      llvm::seq<int64_t>(0, getResourceOperands().size()));
}

//===----------------------------------------------------------------------===//
// stream.channel.default
//===----------------------------------------------------------------------===//

void ChannelDefaultOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel");
}

//===----------------------------------------------------------------------===//
// stream.channel.create
//===----------------------------------------------------------------------===//

void ChannelCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "channel");
}

//===----------------------------------------------------------------------===//
// stream.channel.rank
//===----------------------------------------------------------------------===//

void ChannelRankOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "ccl_rank");
}

//===----------------------------------------------------------------------===//
// stream.channel.count
//===----------------------------------------------------------------------===//

void ChannelCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "ccl_count");
}

//===----------------------------------------------------------------------===//
// stream.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(OpBuilder &builder, OperationState &state,
                         StringRef sym_name) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(sym_name));
}

LogicalResult ExecutableOp::verify() {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// stream.executable.export
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
    for (auto returnOp : getWorkgroupCount().getOps<IREE::Stream::ReturnOp>()) {
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

::mlir::func::FuncOp ExecutableExportOp::lookupFunctionRef() {
  auto executableOp =
      this->getOperation()->getParentOfType<IREE::Stream::ExecutableOp>();
  if (!executableOp) return {};
  return executableOp.getInnerModule().lookupSymbol<::mlir::func::FuncOp>(
      getFunctionRef());
}

//===----------------------------------------------------------------------===//
// stream.binding.subspan
//===----------------------------------------------------------------------===//

LogicalResult BindingSubspanOp::verify() {
  BindingSubspanOp op = *this;
  if (auto shapedType = op.getType().dyn_cast<ShapedType>()) {
    if (failed(verifyOpDynamicDims(op, shapedType, op.getDynamicDims()))) {
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// stream.return
//===----------------------------------------------------------------------===//

MutableOperandRange ReturnOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  return getOperandsMutable();
}

//===----------------------------------------------------------------------===//
// stream.yield
//===----------------------------------------------------------------------===//

MutableOperandRange YieldOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  return getResourceOperandsMutable();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamOps.cpp.inc"  // IWYU pragma: keep
