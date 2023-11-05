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
// custom<ParameterReference>($scope, $key)
//===----------------------------------------------------------------------===//

static ParseResult parseParameterReference(OpAsmParser &parser,
                                           StringAttr &scopeAttr,
                                           StringAttr &keyAttr) {
  auto builder = parser.getBuilder();
  StringAttr firstAttr;
  if (failed(parser.parseCustomAttributeWithFallback(firstAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  if (failed(parser.parseOptionalColon())) {
    keyAttr = firstAttr;
    return success();
  }
  scopeAttr = firstAttr;
  if (failed(parser.parseColon()) ||
      failed(parser.parseCustomAttributeWithFallback(keyAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  return success();
}

static void printParameterReference(OpAsmPrinter &p, Operation *op,
                                    StringAttr scopeAttr, StringAttr keyAttr) {
  if (scopeAttr) {
    p << "\"" << scopeAttr.getValue() << "\"";
    p << "::";
  }
  p << "\"" << keyAttr.getValue() << "\"";
}

//===----------------------------------------------------------------------===//
// io_parameters.load
//===----------------------------------------------------------------------===//

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  // TODO(benvanik): fold hal.buffer.subspan on the result into parameters.
}

//===----------------------------------------------------------------------===//
// io_parameters.read
//===----------------------------------------------------------------------===//

namespace {

struct FoldReadOpTargetBufferSubspan : public OpRewritePattern<ReadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReadOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceOffset = llvm::cast<Value>(op.getSourceOffset());
    auto newTargetBuffer = op.getTargetBuffer();
    auto newTargetOffset = llvm::cast<Value>(op.getTargetOffset());
    if (auto subspanOp = dyn_cast_or_null<IREE::HAL::BufferSubspanOp>(
            newTargetBuffer.getDefiningOp())) {
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(), newSourceOffset);
      newTargetBuffer = subspanOp.getSourceBuffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(), newTargetOffset);
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate)
      return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceOffsetMutable().assign(newSourceOffset);
      op.getTargetBufferMutable().assign(newTargetBuffer);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

} // namespace

void ReadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  // TODO(benvanik): fold hal.buffer.subspan on the result into parameters.
  results.insert<FoldReadOpTargetBufferSubspan>(context);
}

//===----------------------------------------------------------------------===//
// io_parameters.write
//===----------------------------------------------------------------------===//

namespace {

struct FoldWriteOpSourceBufferSubspan : public OpRewritePattern<WriteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(WriteOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = llvm::cast<Value>(op.getSourceOffset());
    auto newTargetOffset = llvm::cast<Value>(op.getTargetOffset());
    if (auto subspanOp = dyn_cast_or_null<IREE::HAL::BufferSubspanOp>(
            newSourceBuffer.getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(), newSourceOffset);
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(),
          rewriter.createOrFold<mlir::arith::IndexCastOp>(
              subspanOp.getLoc(), rewriter.getI64Type(),
              subspanOp.getSourceOffset()),
          newTargetOffset);
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate)
      return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

} // namespace

void WriteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<FoldWriteOpSourceBufferSubspan>(context);
}

//===----------------------------------------------------------------------===//
// custom<ParameterGatherOperations>(
//     $source_scope, $source_keys, $source_offsets,
//     $target_buffer, type($target_buffer), $target_offsets, $target_lengths)
//===----------------------------------------------------------------------===//

static ParseResult parseParameterGatherOperations(
    OpAsmParser &parser, StringAttr &sourceScopeAttr, ArrayAttr &sourceKeysAttr,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sourceOffsets,
    OpAsmParser::UnresolvedOperand &targetBuffer, Type &targetType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetLengths) {
  auto builder = parser.getBuilder();
  SmallVector<Attribute> sourceKeyAttrs;
  do {
    StringAttr rowSourceScopeAttr;
    StringAttr sourceKeyAttr;
    OpAsmParser::UnresolvedOperand sourceOffset;
    OpAsmParser::UnresolvedOperand targetOffset;
    OpAsmParser::UnresolvedOperand targetLength;
    OpAsmParser::UnresolvedOperand rowTargetBuffer;
    Type rowTargetType;
    if (failed(parseParameterReference(parser, rowSourceScopeAttr,
                                       sourceKeyAttr)) ||
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
    if (!targetType) {
      sourceScopeAttr = rowSourceScopeAttr;
      targetBuffer = rowTargetBuffer;
      targetType = rowTargetType;
    } else if (rowSourceScopeAttr != sourceScopeAttr ||
               rowTargetBuffer.name != targetBuffer.name ||
               rowTargetType != targetType) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "each operation must use the same scope and target resource");
    }
    sourceKeyAttrs.push_back(sourceKeyAttr);
    sourceOffsets.push_back(sourceOffset);
    targetOffsets.push_back(targetOffset);
    targetLengths.push_back(targetLength);
  } while (succeeded(parser.parseOptionalComma()));
  sourceKeysAttr = builder.getArrayAttr(sourceKeyAttrs);
  return success();
}

static void printParameterGatherOperations(
    OpAsmPrinter &p, Operation *op, StringAttr sourceScopeAttr,
    ArrayAttr sourceKeysAttr, ValueRange sourceOffsets, Value targetBuffer,
    Type targetType, ValueRange targetOffsets, ValueRange targetLengths) {
  p.increaseIndent();
  p.printNewline();
  llvm::interleave(
      llvm::zip_equal(sourceKeysAttr.getAsRange<StringAttr>(), sourceOffsets,
                      targetOffsets, targetLengths),
      [&](std::tuple<StringAttr, Value, Value, Value> it) {
        auto [sourceKeyAttr, sourceOffset, targetOffset, targetLength] = it;
        printParameterReference(p, op, sourceScopeAttr, sourceKeyAttr);
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
    StringAttr &targetScopeAttr, ArrayAttr &targetKeysAttr,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &targetOffsets) {
  auto builder = parser.getBuilder();
  SmallVector<Attribute> targetKeyAttrs;
  do {
    OpAsmParser::UnresolvedOperand sourceOffset;
    OpAsmParser::UnresolvedOperand sourceLength;
    OpAsmParser::UnresolvedOperand rowSourceBuffer;
    Type rowSourceType;
    StringAttr rowTargetScopeAttr;
    StringAttr targetKeyAttr;
    OpAsmParser::UnresolvedOperand targetOffset;
    if (failed(parser.parseOperand(rowSourceBuffer)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(sourceOffset)) ||
        failed(parser.parseKeyword("for")) ||
        failed(parser.parseOperand(sourceLength)) ||
        failed(parser.parseRSquare()) ||
        failed(parser.parseColonType(rowSourceType)) ||
        failed(parser.parseArrow()) ||
        failed(parseParameterReference(parser, rowTargetScopeAttr,
                                       targetKeyAttr)) ||
        failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(targetOffset)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
    if (!sourceType) {
      sourceBuffer = rowSourceBuffer;
      sourceType = rowSourceType;
      targetScopeAttr = rowTargetScopeAttr;
    } else if (rowSourceBuffer.name != sourceBuffer.name ||
               rowSourceType != sourceType ||
               rowTargetScopeAttr != targetScopeAttr) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "each operation must use the same source resource and scope");
    }
    sourceOffsets.push_back(sourceOffset);
    sourceLengths.push_back(sourceLength);
    targetKeyAttrs.push_back(targetKeyAttr);
    targetOffsets.push_back(targetOffset);
  } while (succeeded(parser.parseOptionalComma()));
  targetKeysAttr = builder.getArrayAttr(targetKeyAttrs);
  return success();
}

static void printParameterScatterOperations(OpAsmPrinter &p, Operation *op,
                                            Value sourceBuffer, Type sourceType,
                                            ValueRange sourceOffsets,
                                            ValueRange sourceLengths,
                                            StringAttr targetScopeAttr,
                                            ArrayAttr targetKeysAttr,
                                            ValueRange targetOffsets) {
  p.increaseIndent();
  p.printNewline();
  llvm::interleave(
      llvm::zip_equal(sourceOffsets, sourceLengths,
                      targetKeysAttr.getAsRange<StringAttr>(), targetOffsets),
      [&](std::tuple<Value, Value, StringAttr, Value> it) {
        auto [sourceOffset, sourceLength, targetKeyAttr, targetOffset] = it;
        p.printOperand(sourceBuffer);
        p << "[";
        p.printOperand(sourceOffset);
        p << " for ";
        p.printOperand(sourceLength);
        p << "] : ";
        p.printType(sourceType);
        p << " -> ";
        printParameterReference(p, op, targetScopeAttr, targetKeyAttr);
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
