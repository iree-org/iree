// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/BuiltinRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

static Value castToI64(Value value, OpBuilder &builder) {
  if (value.getType().isInteger(64)) {
    return value;
  }
  return builder.createOrFold<IREE::VM::ExtI32I64UOp>(
      value.getLoc(), builder.getI64Type(), value);
}

static constexpr const char kItoaBuiltinName[] = "__iree_string_itoa_u64";

// Populates the body of the __iree_string_itoa_u64 builtin helper.
// Converts a uint64 value to its unsigned decimal string representation
// stored in a newly-allocated VM buffer.
//
// The function has signature (i64) -> !vm.ref<!vm.buffer> and is structured as:
//   entry: check for zero, branch to zero_case or loop
//   zero_case: store '0', clone 1 byte, return
//   loop(remaining, position): extract digit, store, divide, cond_br
//   extract: clone the used portion, return
static void buildItoaBody(IREE::VM::FuncOp funcOp, OpBuilder &builder) {
  auto location = funcOp.getLoc();
  auto i32Type = builder.getI32Type();
  auto i64Type = builder.getI64Type();
  auto bufferRefType =
      IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext()));

  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  Value inputValue = entryBlock->getArgument(0);

  Value constantZeroI64 = IREE::VM::ConstI64ZeroOp::create(builder, location);
  Value constantOneI32 = IREE::VM::ConstI32Op::create(builder, location, 1);
  Value constantOneI64 = IREE::VM::ConstI64Op::create(builder, location, 1);
  Value constantTenI64 = IREE::VM::ConstI64Op::create(builder, location, 10);
  Value constantTwentyI64 = IREE::VM::ConstI64Op::create(builder, location, 20);
  Value constantNineteenI64 =
      IREE::VM::ConstI64Op::create(builder, location, 19);
  Value asciiZeroI32 = IREE::VM::ConstI32Op::create(builder, location, 48);

  // Allocate 20-byte scratch buffer (max digits for uint64).
  Value scratchBuffer = IREE::VM::BufferAllocOp::create(
      builder, location, bufferRefType, constantTwentyI64, constantOneI32);

  // Check for zero.
  Value isZero = IREE::VM::CmpEQI64Op::create(builder, location, i32Type,
                                              inputValue, constantZeroI64);

  // Create blocks.
  Block *zeroCaseBlock = funcOp.addBlock();
  Block *loopBlock = funcOp.addBlock();
  loopBlock->addArgument(i64Type, location); // remaining
  loopBlock->addArgument(i64Type, location); // write position
  Block *extractBlock = funcOp.addBlock();
  extractBlock->addArgument(i64Type, location); // start position

  // Entry: cond_br to zero case or loop.
  IREE::VM::CondBranchOp::create(builder, location, isZero, zeroCaseBlock,
                                 ValueRange{}, loopBlock,
                                 ValueRange{inputValue, constantNineteenI64});

  // Zero case: store ASCII '0' at position 0, clone 1 byte, return.
  builder.setInsertionPointToEnd(zeroCaseBlock);
  IREE::VM::BufferStoreI8Op::create(builder, location, scratchBuffer,
                                    constantZeroI64, asciiZeroI32);
  Value zeroResult = IREE::VM::BufferCloneOp::create(
      builder, location, bufferRefType, scratchBuffer, constantZeroI64,
      constantOneI64, constantOneI32);
  IREE::VM::ReturnOp::create(builder, location, ValueRange{zeroResult});

  // Loop: extract digit, store, advance.
  builder.setInsertionPointToEnd(loopBlock);
  Value remaining = loopBlock->getArgument(0);
  Value writePosition = loopBlock->getArgument(1);

  // digit = remaining % 10
  Value digit = IREE::VM::RemI64UOp::create(builder, location, i64Type,
                                            remaining, constantTenI64);
  // Truncate digit to i32 for ASCII conversion.
  Value digitI32 =
      IREE::VM::TruncI64I32Op::create(builder, location, i32Type, digit);
  // ascii = digit + '0'
  Value asciiDigit = IREE::VM::AddI32Op::create(builder, location, i32Type,
                                                digitI32, asciiZeroI32);
  // Store the digit byte.
  IREE::VM::BufferStoreI8Op::create(builder, location, scratchBuffer,
                                    writePosition, asciiDigit);
  // remaining = remaining / 10
  Value nextRemaining = IREE::VM::DivI64UOp::create(builder, location, i64Type,
                                                    remaining, constantTenI64);
  // position = position - 1
  Value nextPosition = IREE::VM::SubI64Op::create(
      builder, location, i64Type, writePosition, constantOneI64);
  // Check if done (remaining == 0).
  Value loopDone = IREE::VM::CmpEQI64Op::create(builder, location, i32Type,
                                                nextRemaining, constantZeroI64);
  IREE::VM::CondBranchOp::create(builder, location, loopDone, extractBlock,
                                 ValueRange{writePosition}, loopBlock,
                                 ValueRange{nextRemaining, nextPosition});

  // Extract: clone from start position to end of scratch buffer.
  builder.setInsertionPointToEnd(extractBlock);
  Value startPosition = extractBlock->getArgument(0);
  Value stringLength = IREE::VM::SubI64Op::create(
      builder, location, i64Type, constantTwentyI64, startPosition);
  Value resultBuffer = IREE::VM::BufferCloneOp::create(
      builder, location, bufferRefType, scratchBuffer, startPosition,
      stringLength, constantOneI32);
  IREE::VM::ReturnOp::create(builder, location, ValueRange{resultBuffer});
}

// Registers all builtin helpers needed by string conversion patterns.
static void registerStringBuiltins(BuiltinRegistry &builtins,
                                   MLIRContext *context) {
  auto i64Type = IntegerType::get(context, 64);
  auto bufferRefType =
      IREE::VM::RefType::get(IREE::VM::BufferType::get(context));
  auto itoaType = FunctionType::get(context, {i64Type}, {bufferRefType});
  builtins.add(kItoaBuiltinName, itoaType, buildItoaBody);
}

struct StringItoaOpConversion
    : public OpConversionPattern<IREE::Util::StringItoaOp> {
  BuiltinRegistry &builtins;
  StringItoaOpConversion(const TypeConverter &typeConverter,
                         MLIRContext *context, BuiltinRegistry &builtins)
      : OpConversionPattern(typeConverter, context), builtins(builtins) {}
  LogicalResult
  matchAndRewrite(IREE::Util::StringItoaOp itoaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto itoaFunc = builtins.use(kItoaBuiltinName);
    Value inputI64 = castToI64(adaptor.getValue(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(itoaOp, itoaFunc,
                                                  ValueRange{inputI64});
    return success();
  }
};

struct StringFormatOpConversion
    : public OpConversionPattern<IREE::Util::StringFormatOp> {
  BuiltinRegistry &builtins;
  StringFormatOpConversion(const TypeConverter &typeConverter,
                           MLIRContext *context, BuiltinRegistry &builtins)
      : OpConversionPattern(typeConverter, context), builtins(builtins) {}
  LogicalResult
  matchAndRewrite(IREE::Util::StringFormatOp formatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto location = formatOp.getLoc();
    auto i64Type = rewriter.getI64Type();
    auto bufferRefType =
        getTypeConverter()->convertType(formatOp.getResult().getType());

    StringRef format = formatOp.getFormat();

    // Parse the format string into segments at compile time.
    struct Segment {
      enum Kind { Literal, Arg };
      Kind kind;
      std::string literal;
      unsigned argIndex;
    };
    SmallVector<Segment> segments;
    unsigned nextSequentialIndex = 0;
    size_t position = 0;
    std::string currentLiteral;
    while (position < format.size()) {
      char character = format[position];
      if (character == '{') {
        if (position + 1 < format.size() && format[position + 1] == '{') {
          currentLiteral += '{';
          position += 2;
          continue;
        }
        if (!currentLiteral.empty()) {
          segments.push_back({Segment::Literal, currentLiteral, 0});
          currentLiteral.clear();
        }
        size_t closePosition = format.find('}', position + 1);
        StringRef content = format.slice(position + 1, closePosition);
        if (content.empty()) {
          segments.push_back({Segment::Arg, {}, nextSequentialIndex++});
        } else {
          unsigned index;
          content.getAsInteger(10, index);
          segments.push_back({Segment::Arg, {}, index});
        }
        position = closePosition + 1;
      } else if (character == '}') {
        if (position + 1 < format.size() && format[position + 1] == '}') {
          currentLiteral += '}';
          position += 2;
          continue;
        }
        ++position;
      } else {
        currentLiteral += character;
        ++position;
      }
    }
    if (!currentLiteral.empty()) {
      segments.push_back({Segment::Literal, currentLiteral, 0});
    }

    Value constantZeroI64 =
        IREE::VM::ConstI64ZeroOp::create(rewriter, location);
    Value constantOneI32 = IREE::VM::ConstI32Op::create(rewriter, location, 1);

    // Phase 1: Convert all integer args to buffers via itoa.
    // Build a vector of buffer values for each arg.
    SmallVector<Value> argBuffers(formatOp.getArgs().size());
    IREE::VM::FuncOp itoaFunc;
    for (auto [index, arg] : llvm::enumerate(adaptor.getArgs())) {
      auto originalArgType = formatOp.getArgs()[index].getType();
      if (isa<IREE::Util::BufferType>(originalArgType)) {
        // Already a buffer (type-converted to vm.ref by adaptor).
        argBuffers[index] = arg;
      } else {
        // Integer arg: call itoa to produce a buffer.
        if (!itoaFunc) {
          itoaFunc = builtins.use(kItoaBuiltinName);
        }
        Value inputI64 = castToI64(arg, rewriter);
        auto callOp = IREE::VM::CallOp::create(rewriter, location, itoaFunc,
                                               ValueRange{inputI64});
        argBuffers[index] = callOp.getResult(0);
      }
    }

    // Phase 2: Compute total length.
    // Literal segment lengths are known at compile time. Buffer segment
    // lengths require vm.buffer.length calls at runtime.
    int64_t staticLength = 0;
    SmallVector<Value> dynamicLengths;
    for (const auto &segment : segments) {
      if (segment.kind == Segment::Literal) {
        staticLength += segment.literal.size();
      } else {
        Value bufferLength = IREE::VM::BufferLengthOp::create(
            rewriter, location, i64Type, argBuffers[segment.argIndex]);
        dynamicLengths.push_back(bufferLength);
      }
    }
    Value totalLength =
        IREE::VM::ConstI64Op::create(rewriter, location, staticLength);
    for (Value dynamicLength : dynamicLengths) {
      totalLength = IREE::VM::AddI64Op::create(rewriter, location, i64Type,
                                               totalLength, dynamicLength);
    }

    // Phase 3: Allocate result buffer.
    Value resultBuffer = IREE::VM::BufferAllocOp::create(
        rewriter, location, bufferRefType, totalLength, constantOneI32);

    // Phase 4: Copy each segment into the result buffer.
    Value currentOffset = constantZeroI64;
    for (const auto &segment : segments) {
      if (segment.kind == Segment::Literal) {
        // Create inline rodata for the literal.
        auto literalAttr = rewriter.getStringAttr(segment.literal);
        Value literalBuffer = IREE::VM::RodataInlineOp::create(
            rewriter, location, bufferRefType, literalAttr);
        Value literalLength = IREE::VM::ConstI64Op::create(
            rewriter, location, segment.literal.size());
        IREE::VM::BufferCopyOp::create(rewriter, location, literalBuffer,
                                       constantZeroI64, resultBuffer,
                                       currentOffset, literalLength);
        currentOffset = IREE::VM::AddI64Op::create(
            rewriter, location, i64Type, currentOffset, literalLength);
      } else {
        // Copy the arg buffer into the result.
        Value argBuffer = argBuffers[segment.argIndex];
        Value argLength = IREE::VM::BufferLengthOp::create(rewriter, location,
                                                           i64Type, argBuffer);
        IREE::VM::BufferCopyOp::create(rewriter, location, argBuffer,
                                       constantZeroI64, resultBuffer,
                                       currentOffset, argLength);
        currentOffset = IREE::VM::AddI64Op::create(rewriter, location, i64Type,
                                                   currentOffset, argLength);
      }
    }

    rewriter.replaceOp(formatOp, resultBuffer);
    return success();
  }
};

} // namespace

void populateUtilStringToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    BuiltinRegistry &builtins,
                                    RewritePatternSet &patterns) {
  registerStringBuiltins(builtins, context);

  conversionTarget.addIllegalOp<IREE::Util::StringFormatOp>();
  conversionTarget.addIllegalOp<IREE::Util::StringItoaOp>();

  patterns.insert<StringItoaOpConversion>(typeConverter, context, builtins);
  patterns.insert<StringFormatOpConversion>(typeConverter, context, builtins);
}

} // namespace mlir::iree_compiler
