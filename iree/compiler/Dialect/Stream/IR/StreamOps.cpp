// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"

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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Op utilities used within the stream dialect
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
    if (results.size() != yieldOp.operands().size()) {
      return yieldOp.emitOpError()
             << "yield result count mismatch with parent op";
    }
    for (auto it : llvm::zip(results, yieldOp.operands())) {
      auto outerValue = std::get<0>(it);
      auto innerValue = std::get<1>(it);
      if (outerValue.getType() != innerValue.getType()) {
        return yieldOp.emitOpError()
               << "result type mismatch: expected " << outerValue.getType()
               << " but got " << innerValue.getType();
      }
    }
    for (auto it : llvm::zip(resultSizes, yieldOp.operand_sizes())) {
      auto outerSize = std::get<0>(it);
      auto innerSize = std::get<1>(it);
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
      yieldOp.operandsMutable().erase(i);
      yieldOp.operand_sizesMutable().erase(i);
    }
  }
}

//===----------------------------------------------------------------------===//
// custom<ResourceRegion>($operands, type($operands), $operand_sizes,
//                        type($results), $result_sizes,
//                        $tied_operands, $body)
//===----------------------------------------------------------------------===//

static ParseResult parseResourceRegion(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &operandSizes,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultSizes,
    ArrayAttr &tiedOperands, Region &body) {
  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
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
          failed(parser.parseRegionArgument(regionArgs.back())) ||
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

  if (failed(parser.parseArrow())) return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parseShapedResultList(parser, operands, operandTypes,
                                     operandSizes, resultTypes, resultSizes,
                                     tiedOperands)) ||
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
  return parser.parseRegion(body, regionArgs, operandTypes,
                            /*enableNameShadowing=*/false);
}

static void printResourceRegion(OpAsmPrinter &p, Operation *op,
                                ValueRange operands, TypeRange operandTypes,
                                ValueRange operandSizes, TypeRange resultTypes,
                                ValueRange resultSizes, ArrayAttr tiedOperands,
                                Region &body) {
  p << "(";
  llvm::interleaveComma(
      llvm::zip(operands, body.getArguments()), p, [&](auto it) {
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
  p << ") -> ";
  if (resultTypes.size() != 1) p << "(";
  printShapedResultList(p, op, operands, operandTypes, operandSizes,
                        resultTypes, resultSizes, tiedOperands);
  if (resultTypes.size() != 1) p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// custom<ExplicitResourceRegion>($operands, type($operands), $operand_sizes,
//                                $body)
//===----------------------------------------------------------------------===//

static ParseResult parseExplicitResourceRegion(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &operandSizes, Region &body) {
  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
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
          failed(parser.parseRegionArgument(regionArgs.back())) ||
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
  if (failed(parser.parseRegion(body, regionArgs, operandTypes,
                                /*enableNameShadowing=*/false))) {
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
      llvm::zip(operands, body.getArguments()), p, [&](auto it) {
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
    SmallVectorImpl<OpAsmParser::OperandType> &dynamicSliceSizes,
    SmallVectorImpl<Type> &packedOffsetTypes) {
  auto indexType = parser.getBuilder().getIndexType();
  SmallVector<Attribute> lifetimeRangeValues;
  do {
    if (failed(parser.parseOptionalLSquare())) break;
    IntegerAttr lifetimeStart;
    IntegerAttr lifetimeEnd;
    OpAsmParser::OperandType dynamicSliceSize;
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
    SmallVectorImpl<OpAsmParser::OperandType> &resultSizes, ArrayAttr &values) {
  SmallVector<Attribute> valueAttrs;
  do {
    Type resultType;
    OpAsmParser::OperandType resultSize;
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
// custom<SymbolAlias>($sym_name, $alias)
//===----------------------------------------------------------------------===//
//  @foo            sym_name: @foo, alias: @foo
//  @foo as @bar    sym_name: @bar, alias: @foo

static ParseResult parseSymbolAlias(OpAsmParser &parser, StringAttr &sym_name,
                                    FlatSymbolRefAttr &alias) {
  if (failed(parser.parseAttribute(alias))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("as"))) {
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(sym_name)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    sym_name = StringAttr::get(parser.getContext(), alias.getValue());
  }
  return success();
}

static void printSymbolAlias(OpAsmPrinter &p, Operation *op,
                             StringAttr sym_name, FlatSymbolRefAttr alias) {
  p.printAttributeWithoutType(alias);
  if (sym_name.getValue() != alias.getValue()) {
    p << " as(\"";
    p.printSymbolName(sym_name.getValue());
    p << "\")";
  }
}

//===----------------------------------------------------------------------===//
// stream.resource.alloc
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(ResourceAllocOp op) {
  if (failed(verifyOpValueSizes(op, op.results(), op.storage_sizes()))) {
    return failure();
  }

  // All allocated resources must have the same lifetime.
  auto anyType = op.results().front().getType();
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

static LogicalResult verifyOp(ResourceMapOp op) {
  if (failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.try_map
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(ResourceTryMapOp op) {
  if (failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.load
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(ResourceLoadOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.store
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(ResourceStoreOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
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

static LogicalResult verifyOp(ResourcePackOp op) {
  size_t sliceCount = op.packed_offsets().size();
  if (op.lifetime_intervals().size() != sliceCount * 2) {
    return op.emitOpError() << "requires a [start, end] range for each slice";
  }
  if (op.dynamic_slice_sizes().size() != sliceCount) {
    return op.emitOpError() << "requires a size for each slice";
  }
  return success();
}

SmallVector<ResourcePackOp::Slice> ResourcePackOp::getSlices() {
  auto intervalPairs = lifetime_intervals().getValue();
  auto sizes = dynamic_slice_sizes();
  auto offsets = packed_offsets();
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

static LogicalResult verifyOp(ResourceConstantsOp op) {
  size_t count = op.results().size();
  if (op.result_sizes().size() != count || op.values().size() != count) {
    return op.emitOpError() << "mismatched constant/result counts";
  }

  // All resources must have the same lifetime.
  auto anyType = op.results().front().getType();
  for (auto result : op.results()) {
    if (result.getType() != anyType) {
      return op.emitError()
             << "all constant resources must have the same lifetime";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// stream.resource.subview
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(ResourceSubviewOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

bool ResourceSubviewOp::isMetadata() { return true; }

Value ResourceSubviewOp::getViewSource() { return source(); }

Value ResourceSubviewOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
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

static LogicalResult verifyOp(TensorImportOp op) {
  if (failed(verifyOpDynamicDims(op, op.result_encoding(),
                                 op.result_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

Value TensorImportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
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

static LogicalResult verifyOp(TensorExportOp op) {
  if (failed(verifyOpDynamicDims(op, op.source_encoding(),
                                 op.source_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.source(), op.source_size()))) {
    return failure();
  }
  return success();
}

Value TensorExportOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(source());
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

static LogicalResult verifyOp(TensorSizeOfOp op) {
  if (failed(verifyOpDynamicDims(op, op.encoding(), op.encoding_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.constant
//===----------------------------------------------------------------------===//

void TensorConstantOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(result(), "cst");
}

static LogicalResult verifyOp(TensorConstantOp op) {
  if (failed(verifyOpDynamicDims(op, op.result_encoding(),
                                 op.result_encoding_dims()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.splat
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorSplatOp op) {
  if (failed(verifyOpDynamicDims(op, op.result_encoding(),
                                 op.result_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.clone
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorCloneOp op) {
  // Clones can't change encodings but they can change shape information.
  auto sourceEncoding = op.source_encoding().cast<RankedTensorType>();
  auto resultEncoding = op.result_encoding().cast<RankedTensorType>();
  if (sourceEncoding.getEncoding() != resultEncoding.getEncoding()) {
    return op.emitOpError() << "clones changing tensor encoding from "
                            << sourceEncoding.getEncoding() << " to "
                            << resultEncoding.getEncoding() << "; not allowed";
  }
  if (failed(verifyOpDynamicDims(op, op.source_encoding(),
                                 op.source_encoding_dims())) ||
      failed(verifyOpDynamicDims(op, op.result_encoding(),
                                 op.result_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.slice
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorSliceOp op) {
  if (failed(verifyOpDynamicDims(op, op.source_encoding(),
                                 op.source_encoding_dims())) ||
      failed(verifyOpDynamicDims(op, op.result_encoding(),
                                 op.result_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  auto sourceType = op.source_encoding().cast<ShapedType>();
  if (op.start_indices().size() != sourceType.getRank() ||
      op.lengths().size() != sourceType.getRank()) {
    return op.emitOpError() << "start_indices/lengths rank mismatch";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.update
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorUpdateOp op) {
  if (failed(verifyOpDynamicDims(op, op.update_encoding(),
                                 op.update_encoding_dims())) ||
      failed(verifyOpDynamicDims(op, op.target_encoding(),
                                 op.target_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.update(), op.update_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.target_size()))) {
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
// stream.tensor.fill
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorFillOp op) {
  if (failed(verifyOpDynamicDims(op, op.target_encoding(),
                                 op.target_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.result(), op.target_size()))) {
    return failure();
  }
  return success();
}

Value TensorFillOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
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

static LogicalResult verifyOp(TensorLoadOp op) {
  if (failed(verifyOpDynamicDims(op, op.source_encoding(),
                                 op.source_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.source(), op.source_size()))) {
    return failure();
  }
  auto sourceType = op.source_encoding().cast<ShapedType>();
  if (op.indices().size() != sourceType.getRank()) {
    return op.emitOpError() << "indices rank mismatch";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.tensor.store
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TensorStoreOp op) {
  if (failed(verifyOpDynamicDims(op, op.target_encoding(),
                                 op.target_encoding_dims())) ||
      failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  auto targetType = op.target_encoding().cast<ShapedType>();
  if (op.indices().size() != targetType.getRank()) {
    return op.emitOpError() << "indices rank mismatch";
  }
  return success();
}

Value TensorStoreOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> TensorStoreOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorStoreOp::getTiedResultOperandIndices() {
  return {0};  // target
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
  setNameFn(result(), "cst");
}

static LogicalResult verifyOp(AsyncConstantOp op) {
  if (failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.async.splat
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncSplatOp op) {
  if (failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

bool AsyncSplatOp::preferCloneToConsumers() { return true; }

//===----------------------------------------------------------------------===//
// stream.async.clone
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncCloneOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

bool AsyncCloneOp::preferCloneToConsumers() { return true; }

//===----------------------------------------------------------------------===//
// stream.async.slice
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncSliceOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

bool AsyncSliceOp::isMetadata() { return true; }

//===----------------------------------------------------------------------===//
// stream.async.fill
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncFillOp op) {
  if (failed(verifyOpValueSizes(op, op.result(), op.target_size()))) {
    return failure();
  }
  return success();
}

Value AsyncFillOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> AsyncFillOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncFillOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.async.update
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncUpdateOp op) {
  if (failed(verifyOpValueSizes(op, op.update(), op.update_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.target_size()))) {
    return failure();
  }
  return success();
}

bool AsyncUpdateOp::isMetadata() { return true; }

Value AsyncUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> AsyncUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.async.copy
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncCopyOp op) {
  if (op.source() == op.target()) {
    // If we want to perform memmove-like operations where it's safe to copy
    // overlapping ranges we'll need to emit some runtime checks. We can in
    // many cases statically detect a lack of overlap just based on symbolic
    // offset equality but that requires some analysis we don't have yet.
    return op.emitOpError() << "cannot copy within the same resource (yet)";
  }
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.target_size()))) {
    return failure();
  }
  return success();
}

Value AsyncCopyOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> AsyncCopyOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> AsyncCopyOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// stream.async.transfer
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncTransferOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.result(), op.result_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.async.load
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncLoadOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.async.store
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(AsyncStoreOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

Value AsyncStoreOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(target());
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

static LogicalResult verifyOp(AsyncDispatchOp op) {
  if (failed(verifyOpValueSizes(op, op.operands(), op.operand_sizes())) ||
      failed(verifyOpValueSizes(op, op.results(), op.result_sizes()))) {
    return failure();
  }
  return success();
}

std::pair<unsigned, unsigned> AsyncDispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);  // $operands
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
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(resultSizes.size()),
                         awaitTimepoint ? 1 : 0,
                     }));
  state.addRegion();
}

static LogicalResult verifyOp(AsyncExecuteOp op) {
  if (failed(RegionBranchOpInterface::verifyTypes(op))) return failure();
  if (failed(verifyOpValueSizes(op, op.operands(), op.operand_sizes())) ||
      failed(verifyOpValueSizes(op, op.results(), op.result_sizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.body())) ||
      failed(verifyEscapingResources(op.body(), op.results(),
                                     op.result_sizes()))) {
    return failure();
  }
  return success();
}

std::pair<unsigned, unsigned> AsyncExecuteOp::getTiedResultsIndexAndLength() {
  return {0, results().size()};
}

OperandRange AsyncExecuteOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");
  return operands();
}

void AsyncExecuteOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(results()));
  } else {
    regions.push_back(RegionSuccessor(&body(), body().getArguments()));
  }
}

Operation::operand_range AsyncExecuteOp::getClosureOperands() {
  return operands();
}

Operation::result_range AsyncExecuteOp::getClosureResults() {
  return results();
}

bool AsyncExecuteOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess AsyncExecuteOp::getOperandAccess(
    unsigned operandIndex) {
  auto arg = body().getArgument(operandIndex);
  return computeValueAccess(arg);
}

IREE::Util::ValueAccess AsyncExecuteOp::getResultAccess(unsigned resultIndex) {
  auto yieldOp = cast<YieldOp>(body().getBlocks().front().getTerminator());
  return computeValueAccess(yieldOp.getOperand(resultIndex));
}

IREE::Util::ClosureOpInterface
AsyncExecuteOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  auto newResultTypes = llvm::to_vector<4>(
      llvm::map_range(results(), [](auto value) { return value.getType(); }));
  auto newResultSizes = llvm::to_vector<4>(result_sizes());
  auto newOperandsValues = llvm::to_vector<4>(operands());
  auto newOperandSizes = llvm::to_vector<4>(operand_sizes());
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
      getLoc(), newResultTypes, newResultSizes, await_timepoint(),
      newOperandsValues, newOperandSizes, newTiedOperandIndices,
      getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  eraseStreamRegionResults(newBody, excludedResultIndices);
  newBody.front().eraseArguments(excludedOperandIndices);
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
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(resultSizes.size()),
                     }));
  state.addRegion();
}

static LogicalResult verifyOp(AsyncConcurrentOp op) {
  if (failed(RegionBranchOpInterface::verifyTypes(op))) return failure();
  if (failed(verifyOpValueSizes(op, op.operands(), op.operand_sizes())) ||
      failed(verifyOpValueSizes(op, op.results(), op.result_sizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.body())) ||
      failed(verifyEscapingResources(op.body(), op.results(),
                                     op.result_sizes()))) {
    return failure();
  }
  return success();
}

OperandRange AsyncConcurrentOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");
  return operands();
}

void AsyncConcurrentOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(results()));
  } else {
    regions.push_back(RegionSuccessor(&body(), body().getArguments()));
  }
}

Operation::operand_range AsyncConcurrentOp::getClosureOperands() {
  return operands();
}

Operation::result_range AsyncConcurrentOp::getClosureResults() {
  return results();
}

bool AsyncConcurrentOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess AsyncConcurrentOp::getOperandAccess(
    unsigned operandIndex) {
  auto arg = body().getArgument(operandIndex);
  return computeValueAccess(arg);
}

IREE::Util::ValueAccess AsyncConcurrentOp::getResultAccess(
    unsigned resultIndex) {
  auto yieldOp = cast<YieldOp>(body().getBlocks().front().getTerminator());
  return computeValueAccess(yieldOp.getOperand(resultIndex));
}

IREE::Util::ClosureOpInterface
AsyncConcurrentOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices, PatternRewriter &rewriter) {
  auto newResultTypes = llvm::to_vector<4>(getResultTypes());
  auto newResultSizes = llvm::to_vector<4>(result_sizes());
  auto newOperandsValues = llvm::to_vector<4>(operands());
  auto newOperandSizes = llvm::to_vector<4>(operand_sizes());
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
  newBody.front().eraseArguments(excludedOperandIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// stream.cmd.flush
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdFlushOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.invalidate
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdInvalidateOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.discard
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdDiscardOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.fill
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdFillOp op) {
  if (failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.copy
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdCopyOp op) {
  if (failed(verifyOpValueSizes(op, op.source(), op.source_size())) ||
      failed(verifyOpValueSizes(op, op.target(), op.target_size()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// stream.cmd.dispatch
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdDispatchOp op) {
  size_t resourceCount = op.resources().size();
  if (op.resource_sizes().size() != resourceCount ||
      op.resource_offsets().size() != resourceCount ||
      op.resource_lengths().size() != resourceCount ||
      op.resource_accesses().size() != resourceCount) {
    return op->emitOpError() << "dispatch with " << resourceCount
                             << " resources has mismatched associated ranges";
  }
  return success();
}

static ParseResult parseDispatchResources(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &resources,
    SmallVectorImpl<Type> &resourceTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resourceSizes,
    SmallVectorImpl<OpAsmParser::OperandType> &resourceOffsets,
    SmallVectorImpl<OpAsmParser::OperandType> &resourceLengths,
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
    if (bitEnumContains(resourceAccess,
                        IREE::Stream::ResourceAccessBitfield::Read) &&
        bitEnumContains(resourceAccess,
                        IREE::Stream::ResourceAccessBitfield::Write)) {
      p << "rw";
    } else if (bitEnumContains(resourceAccess,
                               IREE::Stream::ResourceAccessBitfield::Read)) {
      p << "ro";
    } else if (bitEnumContains(resourceAccess,
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
SmallVector<unsigned> CmdDispatchOp::makeOperandToArgMap(mlir::FuncOp funcOp) {
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
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         awaitTimepoint ? 1 : 0,
                     }));
  state.addRegion();
}

// Returns success if the given op is a known valid stream.cmd.* op for use
// within an execution region.
static LogicalResult verifyCmdOp(Operation *op) {
  // TODO(benvanik): add a trait that lets us avoid this switch.
  if (!TypeSwitch<Operation *, bool>(op)
           .Case<IREE::Stream::CmdFlushOp, IREE::Stream::CmdInvalidateOp,
                 IREE::Stream::CmdDiscardOp, IREE::Stream::CmdFillOp,
                 IREE::Stream::CmdCopyOp, IREE::Stream::CmdDispatchOp,
                 IREE::Stream::CmdSerialOp, IREE::Stream::CmdConcurrentOp>(
               [](auto op) { return true; })
           .Case<IREE::Stream::YieldOp>([](auto op) { return true; })
           .Default(false)) {
    return op->emitOpError()
           << "explicit execution regions must only contain explicit ops";
  }
  return success();
}

static LogicalResult verifyOp(CmdExecuteOp op) {
  if (failed(RegionBranchOpInterface::verifyTypes(op))) return failure();
  if (failed(verifyOpValueSizes(op, op.operands(), op.operand_sizes()))) {
    return failure();
  }
  if (failed(verifyAllResourcesCaptured(op.body()))) {
    return failure();
  }
  for (auto &nestedOp : op.body().front()) {
    if (failed(verifyCmdOp(&nestedOp))) return failure();
  }
  return success();
}

OperandRange CmdExecuteOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");
  return operands();
}

void CmdExecuteOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Unconditional control flow into the region and back to the parent, so
  // return the correct RegionSuccessor purely based on the index being None or
  // 0.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&body(), body().getArguments()));
  }
}

Operation::operand_range CmdExecuteOp::getClosureOperands() {
  return operands();
}

Operation::result_range CmdExecuteOp::getClosureResults() {
  return Operation::result_range(nullptr, 0);
}

bool CmdExecuteOp::canClosureContainOp(Operation *op) { return false; }

IREE::Util::ValueAccess CmdExecuteOp::getOperandAccess(unsigned operandIndex) {
  auto arg = body().getArgument(operandIndex);
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
  auto newOperandsValues = llvm::to_vector<4>(operands());
  auto newOperandSizes = llvm::to_vector<4>(operand_sizes());
  IREE::Util::excludeClosureOperandsAndResults(
      newOperandsValues, newOperandSizes, excludedOperandIndices,
      newResultTypes, newResultSizes, excludedResultIndices);

  auto newOp = rewriter.create<CmdExecuteOp>(getLoc(), await_timepoint(),
                                             newOperandsValues, newOperandSizes,
                                             getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  newBody.front().eraseArguments(excludedOperandIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// stream.cmd.serial
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdSerialOp op) {
  for (auto &nestedOp : op.body().front()) {
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
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&body(), {}));
  }
}

//===----------------------------------------------------------------------===//
// stream.cmd.concurrent
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(CmdConcurrentOp op) {
  for (auto &nestedOp : op.body().front()) {
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
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor({}));
  } else {
    regions.push_back(RegionSuccessor(&body(), {}));
  }
}

//===----------------------------------------------------------------------===//
// stream.timepoint.join
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(TimepointJoinOp op) {
  // We could test if timepoints all come from the same place - this is not
  // strictly required but if we could avoid it things will be easier to
  // implement at runtime (won't have to do a cuda<->vulkan sync, etc).
  return success();
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
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandSizes.size()),
                         static_cast<int32_t>(1),  // timepoint
                     }));
}

static LogicalResult verifyOp(TimepointAwaitOp op) {
  if (failed(verifyOpValueSizes(op, op.operands(), op.operand_sizes())) ||
      failed(verifyOpValueSizes(op, op.results(), op.operand_sizes()))) {
    return failure();
  }
  return success();
}

::llvm::Optional<unsigned> TimepointAwaitOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {resultIndex};
}

SmallVector<int64_t, 4> TimepointAwaitOp::getTiedResultOperandIndices() {
  return llvm::to_vector<4>(llvm::seq<int64_t>(0, operands().size()));
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

static LogicalResult verifyOp(ExecutableOp op) {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// stream.executable.entry
//===----------------------------------------------------------------------===//

void ExecutableExportOp::build(OpBuilder &builder, OperationState &state,
                               StringRef sym_name,
                               FlatSymbolRefAttr function_ref) {
  build(builder, state, /*sym_visibility=*/nullptr,
        builder.getStringAttr(sym_name), function_ref);
}

//===----------------------------------------------------------------------===//
// stream.binding.subspan
//===----------------------------------------------------------------------===//

static LogicalResult verifyOp(BindingSubspanOp op) {
  if (auto shapedType = op.getType().dyn_cast<ShapedType>()) {
    if (failed(verifyOpDynamicDims(op, shapedType, op.dynamic_dims()))) {
      return failure();
    }
  }

  return success();
}

Value BindingSubspanOp::buildOperandRankedShape(unsigned idx,
                                                OpBuilder &builder) {
  return {};
}

Value BindingSubspanOp::buildResultRankedShape(unsigned idx,
                                               OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), dynamic_dims(),
                                         builder);
}

//===----------------------------------------------------------------------===//
// stream.yield
//===----------------------------------------------------------------------===//

MutableOperandRange YieldOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  return operandsMutable();
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
