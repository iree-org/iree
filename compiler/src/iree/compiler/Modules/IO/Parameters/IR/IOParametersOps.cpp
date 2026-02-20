// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

//===----------------------------------------------------------------------===//
// Shared parameter reference parsing/printing: %scope :: %key  or  %key
//===----------------------------------------------------------------------===//

// Parses an operand, then checks for `::`. If present, the first operand was
// the scope and the second is the key. Otherwise the first operand is the key
// and scope is unset.
static ParseResult
parseParameterReference(OpAsmParser &parser,
                        std::optional<OpAsmParser::UnresolvedOperand> &scope,
                        OpAsmParser::UnresolvedOperand &key) {
  OpAsmParser::UnresolvedOperand firstOperand;
  if (failed(parser.parseOperand(firstOperand))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalColon())) {
    scope = firstOperand;
    if (failed(parser.parseColon()) || failed(parser.parseOperand(key))) {
      return failure();
    }
  } else {
    key = firstOperand;
  }
  return success();
}

static void printParameterReference(OpAsmPrinter &p, Operation *op, Value scope,
                                    Value key) {
  if (scope) {
    p.printOperand(scope);
    p << "::";
  }
  p.printOperand(key);
}

//===----------------------------------------------------------------------===//
// custom<ParameterLoadOperations>(
//     $source_scope, $source_keys, $source_offsets,
//     type($results), $result_sizes)
//===----------------------------------------------------------------------===//

static ParseResult parseParameterLoadOperations(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &sourceScope,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceKeys,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceOffsets,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultSizes) {
  std::optional<OpAsmParser::UnresolvedOperand> firstScope;
  bool firstRow = true;
  do {
    std::optional<OpAsmParser::UnresolvedOperand> rowScope;
    OpAsmParser::UnresolvedOperand sourceKey;
    OpAsmParser::UnresolvedOperand sourceOffset;
    Type resultType;
    OpAsmParser::UnresolvedOperand resultSize;
    if (failed(parseParameterReference(parser, rowScope, sourceKey)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(sourceOffset)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.parseColonType(resultType)) ||
        failed(parser.parseLBrace()) ||
        failed(parser.parseOperand(resultSize)) ||
        failed(parser.parseRBrace())) {
      return failure();
    }
    if (firstRow) {
      firstScope = rowScope;
      sourceScope = rowScope;
      firstRow = false;
    } else if (rowScope.has_value() != firstScope.has_value() ||
               (rowScope.has_value() && rowScope->name != firstScope->name)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "each operation must use the same scope");
    }
    sourceKeys.push_back(sourceKey);
    sourceOffsets.push_back(sourceOffset);
    resultTypes.push_back(resultType);
    resultSizes.push_back(resultSize);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void
printParameterLoadOperations(OpAsmPrinter &p, Operation *op, Value sourceScope,
                             ValueRange sourceKeys, ValueRange sourceOffsets,
                             TypeRange resultTypes, ValueRange resultSizes) {
  p.increaseIndent();
  p.printNewline();
  llvm::interleave(
      llvm::zip_equal(sourceKeys, sourceOffsets, resultTypes, resultSizes),
      [&](std::tuple<Value, Value, Type, Value> it) {
        auto [sourceKey, sourceOffset, resultType, resultSize] = it;
        printParameterReference(p, op, sourceScope, sourceKey);
        p << "[";
        p.printOperand(sourceOffset);
        p << "] : ";
        p.printType(resultType);
        p << "{";
        p.printOperand(resultSize);
        p << "}";
      },
      [&]() {
        p << ',';
        p.printNewline();
      });
  p.decreaseIndent();
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// io_parameters.load
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  LoadOp op = *this;
  size_t expectedCount = op.getSourceKeys().size();
  if (op.getSourceOffsets().size() != expectedCount ||
      op.getLengths().size() != expectedCount) {
    return op.emitOpError() << "requires that the source keys, source offsets, "
                               "and result sizes are all 1:1";
  }
  return success();
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  // TODO(benvanik): fold hal.buffer.subspan on the result into parameters.
}

//===----------------------------------------------------------------------===//
// custom<ParameterGatherOperations>(
//     $source_scope, $source_keys, $source_offsets,
//     $target_buffer, type($target_buffer), $target_offsets, $target_lengths)
//===----------------------------------------------------------------------===//

static ParseResult parseParameterGatherOperations(
    OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &sourceScope,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceKeys,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceOffsets,
    OpAsmParser::UnresolvedOperand &targetBuffer, Type &targetType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetLengths) {
  std::optional<OpAsmParser::UnresolvedOperand> firstScope;
  bool firstRow = true;
  do {
    std::optional<OpAsmParser::UnresolvedOperand> rowScope;
    OpAsmParser::UnresolvedOperand sourceKey;
    OpAsmParser::UnresolvedOperand sourceOffset;
    OpAsmParser::UnresolvedOperand targetOffset;
    OpAsmParser::UnresolvedOperand targetLength;
    OpAsmParser::UnresolvedOperand rowTargetBuffer;
    Type rowTargetType;
    if (failed(parseParameterReference(parser, rowScope, sourceKey)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(sourceOffset)) ||
        failed(parser.parseRSquare()) || failed(parser.parseArrow()) ||
        failed(parser.parseOperand(rowTargetBuffer)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(targetOffset)) ||
        failed(parser.parseKeyword("for")) ||
        failed(parser.parseOperand(targetLength)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.parseColonType(rowTargetType))) {
      return failure();
    }
    if (firstRow) {
      firstScope = rowScope;
      sourceScope = rowScope;
      targetBuffer = rowTargetBuffer;
      targetType = rowTargetType;
      firstRow = false;
    } else if (rowScope.has_value() != firstScope.has_value() ||
               (rowScope.has_value() && rowScope->name != firstScope->name) ||
               rowTargetBuffer.name != targetBuffer.name ||
               rowTargetType != targetType) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "each operation must use the same scope and target resource");
    }
    sourceKeys.push_back(sourceKey);
    sourceOffsets.push_back(sourceOffset);
    targetOffsets.push_back(targetOffset);
    targetLengths.push_back(targetLength);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printParameterGatherOperations(
    OpAsmPrinter &p, Operation *op, Value sourceScope, ValueRange sourceKeys,
    ValueRange sourceOffsets, Value targetBuffer, Type targetType,
    ValueRange targetOffsets, ValueRange targetLengths) {
  p.increaseIndent();
  p.printNewline();
  llvm::interleave(
      llvm::zip_equal(sourceKeys, sourceOffsets, targetOffsets, targetLengths),
      [&](std::tuple<Value, Value, Value, Value> it) {
        auto [sourceKey, sourceOffset, targetOffset, targetLength] = it;
        printParameterReference(p, op, sourceScope, sourceKey);
        p << "[";
        p.printOperand(sourceOffset);
        p << "] -> ";
        p.printOperand(targetBuffer);
        p << "[";
        p.printOperand(targetOffset);
        p << " for ";
        p.printOperand(targetLength);
        p << "] : ";
        p.printType(targetType);
      },
      [&]() {
        p << ',';
        p.printNewline();
      });
  p.decreaseIndent();
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// io_parameters.gather
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  GatherOp op = *this;
  size_t expectedCount = op.getSourceKeys().size();
  if (op.getSourceOffsets().size() != expectedCount ||
      op.getTargetOffsets().size() != expectedCount ||
      op.getTargetLengths().size() != expectedCount) {
    return op.emitOpError() << "requires that the source keys, target offsets, "
                               "and target lengths are all 1:1";
  }
  return success();
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  // TODO(benvanik): find a good way of folding in subspans; tricky because if
  // buffers differ across entries then we can't reassign.
}

//===----------------------------------------------------------------------===//
// custom<ParameterScatterOperations>(
//     $source_buffer, type($source_buffer), $source_offsets, $source_lengths,
//     $target_scope, $target_keys, $target_offsets)
//===----------------------------------------------------------------------===//

static ParseResult parseParameterScatterOperations(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &sourceBuffer,
    Type &sourceType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceLengths,
    std::optional<OpAsmParser::UnresolvedOperand> &targetScope,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetKeys,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetOffsets) {
  std::optional<OpAsmParser::UnresolvedOperand> firstScope;
  bool firstRow = true;
  do {
    OpAsmParser::UnresolvedOperand sourceOffset;
    OpAsmParser::UnresolvedOperand sourceLength;
    OpAsmParser::UnresolvedOperand rowSourceBuffer;
    Type rowSourceType;
    std::optional<OpAsmParser::UnresolvedOperand> rowScope;
    OpAsmParser::UnresolvedOperand targetKey;
    OpAsmParser::UnresolvedOperand targetOffset;
    if (failed(parser.parseOperand(rowSourceBuffer)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(sourceOffset)) ||
        failed(parser.parseKeyword("for")) ||
        failed(parser.parseOperand(sourceLength)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.parseColonType(rowSourceType)) ||
        failed(parser.parseArrow()) ||
        failed(parseParameterReference(parser, rowScope, targetKey)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(targetOffset)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
    if (firstRow) {
      sourceBuffer = rowSourceBuffer;
      sourceType = rowSourceType;
      firstScope = rowScope;
      targetScope = rowScope;
      firstRow = false;
    } else if (rowSourceBuffer.name != sourceBuffer.name ||
               rowSourceType != sourceType ||
               rowScope.has_value() != firstScope.has_value() ||
               (rowScope.has_value() && rowScope->name != firstScope->name)) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "each operation must use the same source resource and scope");
    }
    sourceOffsets.push_back(sourceOffset);
    sourceLengths.push_back(sourceLength);
    targetKeys.push_back(targetKey);
    targetOffsets.push_back(targetOffset);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printParameterScatterOperations(
    OpAsmPrinter &p, Operation *op, Value sourceBuffer, Type sourceType,
    ValueRange sourceOffsets, ValueRange sourceLengths, Value targetScope,
    ValueRange targetKeys, ValueRange targetOffsets) {
  p.increaseIndent();
  p.printNewline();
  llvm::interleave(
      llvm::zip_equal(sourceOffsets, sourceLengths, targetKeys, targetOffsets),
      [&](std::tuple<Value, Value, Value, Value> it) {
        auto [sourceOffset, sourceLength, targetKey, targetOffset] = it;
        p.printOperand(sourceBuffer);
        p << "[";
        p.printOperand(sourceOffset);
        p << " for ";
        p.printOperand(sourceLength);
        p << "] : ";
        p.printType(sourceType);
        p << " -> ";
        printParameterReference(p, op, targetScope, targetKey);
        p << "[";
        p.printOperand(targetOffset);
        p << "]";
      },
      [&]() {
        p << ',';
        p.printNewline();
      });
  p.decreaseIndent();
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// io_parameters.scatter
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  ScatterOp op = *this;
  size_t expectedCount = op.getTargetKeys().size();
  if (op.getSourceOffsets().size() != expectedCount ||
      op.getSourceLengths().size() != expectedCount ||
      op.getTargetOffsets().size() != expectedCount) {
    return op.emitOpError() << "requires that the source offsets, source "
                               "lengths, and target keys are all 1:1";
  }
  return success();
}

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  // TODO(benvanik): find a good way of folding in subspans; tricky because if
  // buffers differ across entries then we can't reassign.
}

} // namespace mlir::iree_compiler::IREE::IO::Parameters

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.cpp.inc"
