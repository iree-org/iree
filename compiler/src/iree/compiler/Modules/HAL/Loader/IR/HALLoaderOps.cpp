// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"

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

namespace mlir::iree_compiler::IREE::HAL::Loader {

//===----------------------------------------------------------------------===//
// custom<DispatchBindings>($binding_buffers,
//                          type($binding_buffers),
//                          $binding_offsets,
//                          $binding_lengths)
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchBindings(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &buffers,
    SmallVectorImpl<Type> &bufferTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &bufferOffsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &bufferLengths) {
  do {
    OpAsmParser::UnresolvedOperand ordinal;
    OpAsmParser::UnresolvedOperand buffer;
    Type bufferType;
    OpAsmParser::UnresolvedOperand bufferOffset;
    OpAsmParser::UnresolvedOperand bufferLength;
    if (failed(parser.parseLParen()) || failed(parser.parseOperand(buffer)) ||
        failed(parser.parseColonType(bufferType)) ||
        failed(parser.parseRParen()) || failed(parser.parseLSquare()) ||
        failed(parser.parseOperand(bufferOffset)) ||
        failed(parser.parseComma()) ||
        failed(parser.parseOperand(bufferLength)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
    buffers.push_back(buffer);
    bufferTypes.push_back(bufferType);
    bufferOffsets.push_back(bufferOffset);
    bufferLengths.push_back(bufferLength);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

static void printDispatchBindings(OpAsmPrinter &p, Operation *op,
                                  ValueRange buffers, TypeRange bufferTypes,
                                  ValueRange bufferOffsets,
                                  ValueRange bufferLengths) {
  llvm::interleaveComma(
      llvm::zip_equal(buffers, bufferTypes, bufferOffsets, bufferLengths), p,
      [&](std::tuple<Value, Type, Value, Value> it) {
        p.printNewline();
        p << "  ";
        p << "(";
        p.printOperand(std::get<0>(it));
        p << " : ";
        p.printType(std::get<1>(it));
        p << ")[";
        p.printOperand(std::get<2>(it));
        p << ", ";
        p.printOperand(std::get<3>(it));
        p << "]";
      });
  p.printNewline();
}

//===----------------------------------------------------------------------===//
// hal_loader.executable.query_support
//===----------------------------------------------------------------------===//

void ExecutableQuerySupportOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getSupported(), (getExecutableFormat() + "_supported").str());
}

//===----------------------------------------------------------------------===//
// hal_loader.executable.load
//===----------------------------------------------------------------------===//

void ExecutableLoadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "exe");
}

//===----------------------------------------------------------------------===//
// hal_loader.executable.lookup
//===----------------------------------------------------------------------===//

void ExecutableLookupOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "exe");
}

LogicalResult
ExecutableLookupOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto exportOp = symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableOp>(
      op, getExecutableAttr());
  if (!exportOp) {
    return op->emitOpError() << "undefined executable: " << getExecutable();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// hal_loader.executable.dispatch
//===----------------------------------------------------------------------===//

static LogicalResult verifyDispatchBindings(Operation *op, OperandRange buffers,
                                            OperandRange offsets,
                                            OperandRange lengths) {
  if (buffers.size() != offsets.size() || buffers.size() != lengths.size()) {
    return op->emitOpError("binding buffers/offsets/lengths must match; have ")
           << buffers.size() << "/" << offsets.size() << "/" << lengths.size();
  }
  return success();
}

LogicalResult ExecutableDispatchSymbolOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto exportOp =
      symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
          op, getEntryPoint());
  if (!exportOp) {
    return op->emitOpError() << "undefined entry point: " << getEntryPoint();
  }
  return verifyDispatchBindings(getOperation(), getBindingBuffers(),
                                getBindingOffsets(), getBindingLengths());
}

LogicalResult ExecutableDispatchOp::verify() {
  return verifyDispatchBindings(getOperation(), getBindingBuffers(),
                                getBindingOffsets(), getBindingLengths());
}

namespace {

// Folds subspan ranges into dispatch resource ranges.
struct FoldBindingSubspansIntoDispatchOp
    : public OpRewritePattern<ExecutableDispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExecutableDispatchOp op,
                                PatternRewriter &rewriter) const override {
    bool didChangeAny = false;
    SmallVector<Value> bindingBuffers;
    SmallVector<Value> bindingOffsets;
    SmallVector<Value> bindingLengths;
    for (auto [bindingBuffer, bindingOffset] :
         llvm::zip_equal(op.getBindingBuffers(), op.getBindingOffsets())) {
      auto subspanOp =
          IREE::Util::BufferSubspanOp::findSubspanOp(bindingBuffer);
      if (!subspanOp) {
        // No subspan, unchanged.
        bindingBuffers.push_back(bindingBuffer);
        bindingOffsets.push_back(bindingOffset);
        continue;
      }
      // Update storage to the source of the subspan and add the subspan offset.
      didChangeAny = true;
      auto fusedLoc = rewriter.getFusedLoc({subspanOp.getLoc(), op.getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, subspanOp.getSourceOffset(), bindingOffset);
      bindingBuffers.push_back(subspanOp.getSource());
      bindingOffsets.push_back(newOffset);
    }
    if (!didChangeAny)
      return failure();
    rewriter.modifyOpInPlace(op, [&]() {
      op.getBindingBuffersMutable().assign(bindingBuffers);
      op.getBindingOffsetsMutable().assign(bindingOffsets);
    });
    return success();
  }
};

} // namespace

void ExecutableDispatchOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldBindingSubspansIntoDispatchOp>(context);
}

} // namespace mlir::iree_compiler::IREE::HAL::Loader

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.cpp.inc"
