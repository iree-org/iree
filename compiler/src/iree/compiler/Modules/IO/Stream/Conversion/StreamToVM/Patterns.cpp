// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Stream/Conversion/StreamToVM/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ReadByteOpConversion
    : public OpConversionPattern<IREE::IO::Stream::ReadByteOp> {
  ReadByteOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                       TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::ReadByteOp readByteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callOp = rewriter.create<IREE::VM::CallOp>(readByteOp.getLoc(),
                                                    importOp.getSymNameAttr(),
                                                    TypeRange{
                                                        rewriter.getI32Type(),
                                                    },
                                                    ValueRange{
                                                        adaptor.getHandle(),
                                                    });
    copyImportAttrs(importOp, callOp);
    Value zero = rewriter.create<IREE::VM::ConstI32ZeroOp>(readByteOp.getLoc());
    Value ltz = rewriter.create<IREE::VM::CmpLTI32SOp>(
        readByteOp.getLoc(), rewriter.getI32Type(), callOp.getResult(0), zero);
    rewriter.replaceOp(readByteOp, {callOp.getResult(0), ltz});
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

struct ExpandReadBytesOpConversion
    : public OpConversionPattern<IREE::IO::Stream::ReadBytesOp> {
  ExpandReadBytesOpConversion(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::ReadBytesOp readBytesOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOffset())
      return failure();
    Value offset =
        rewriter.create<IREE::VM::ConstI64ZeroOp>(readBytesOp.getLoc());
    Value length = rewriter.createOrFold<IREE::VM::BufferLengthOp>(
        readBytesOp.getLoc(), rewriter.getI64Type(), adaptor.getBuffer());
    rewriter.replaceOpWithNewOp<IREE::IO::Stream::ReadBytesOp>(
        readBytesOp, readBytesOp.getResult().getType(), adaptor.getHandle(),
        adaptor.getBuffer(), offset, length);
    return success();
  }
};

struct ReadLineOpConversion
    : public OpConversionPattern<IREE::IO::Stream::ReadLineOp> {
  ReadLineOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::ReadLineOp readLineOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value delimiter =
        rewriter.create<IREE::VM::ConstI32Op>(readLineOp.getLoc(), /*\n*/ 10);
    rewriter.replaceOpWithNewOp<IREE::IO::Stream::ReadDelimiterOp>(
        readLineOp, readLineOp.getResult().getType(), adaptor.getHandle(),
        delimiter);
    return success();
  }
};

struct WriteNewlineOpConversion
    : public OpConversionPattern<IREE::IO::Stream::WriteNewlineOp> {
  WriteNewlineOpConversion(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::WriteNewlineOp writeNewlineOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value delimiter = rewriter.create<IREE::VM::ConstI32Op>(
        writeNewlineOp.getLoc(), /*\n*/ 10);
    rewriter.replaceOpWithNewOp<IREE::IO::Stream::WriteByteOp>(
        writeNewlineOp, adaptor.getHandle(), delimiter);
    return success();
  }
};

struct ExpandWriteBytesOpConversion
    : public OpConversionPattern<IREE::IO::Stream::WriteBytesOp> {
  ExpandWriteBytesOpConversion(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::WriteBytesOp writeBytesOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOffset())
      return failure();
    Value offset =
        rewriter.create<IREE::VM::ConstI64ZeroOp>(writeBytesOp.getLoc());
    Value length = rewriter.createOrFold<IREE::VM::BufferLengthOp>(
        writeBytesOp.getLoc(), rewriter.getI64Type(), adaptor.getBuffer());
    rewriter.replaceOpWithNewOp<IREE::IO::Stream::WriteBytesOp>(
        writeBytesOp, adaptor.getHandle(), adaptor.getBuffer(), offset, length);
    return success();
  }
};

struct WriteLineOpConversion
    : public OpConversionPattern<IREE::IO::Stream::WriteLineOp> {
  WriteLineOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
  LogicalResult
  matchAndRewrite(IREE::IO::Stream::WriteLineOp writeLineOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value offset =
        rewriter.create<IREE::VM::ConstI64ZeroOp>(writeLineOp.getLoc());
    Value length = rewriter.createOrFold<IREE::VM::BufferLengthOp>(
        writeLineOp.getLoc(), rewriter.getI64Type(), adaptor.getBuffer());
    rewriter.create<IREE::IO::Stream::WriteBytesOp>(
        writeLineOp.getLoc(), adaptor.getHandle(), adaptor.getBuffer(), offset,
        length);
    Value delimiter =
        rewriter.create<IREE::VM::ConstI32Op>(writeLineOp.getLoc(), /*\n*/ 10);
    rewriter.create<IREE::IO::Stream::WriteByteOp>(
        writeLineOp.getLoc(), adaptor.getHandle(), delimiter);
    rewriter.eraseOp(writeLineOp);
    return success();
  }
};

} // namespace

void populateIOStreamToVMPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  SymbolTable &importSymbols,
                                  RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::ConsoleStdinOp>>(
      context, importSymbols, typeConverter, "io_stream.console.stdin");
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::ConsoleStdoutOp>>(
      context, importSymbols, typeConverter, "io_stream.console.stdout");
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::ConsoleStderrOp>>(
      context, importSymbols, typeConverter, "io_stream.console.stderr");

  patterns.insert<VMImportOpConversion<IREE::IO::Stream::OffsetOp>>(
      context, importSymbols, typeConverter, "io_stream.offset");
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::LengthOp>>(
      context, importSymbols, typeConverter, "io_stream.length");

  patterns.insert<ReadByteOpConversion>(context, importSymbols, typeConverter,
                                        "io_stream.read.byte");
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::ReadBytesOp>>(
      context, importSymbols, typeConverter, "io_stream.read.bytes");
  patterns.insert<ExpandReadBytesOpConversion>(context);
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::ReadDelimiterOp>>(
      context, importSymbols, typeConverter, "io_stream.read.delimiter");
  patterns.insert<ReadLineOpConversion>(context);

  patterns.insert<VMImportOpConversion<IREE::IO::Stream::WriteByteOp>>(
      context, importSymbols, typeConverter, "io_stream.write.byte");
  patterns.insert<WriteNewlineOpConversion>(context);
  patterns.insert<VMImportOpConversion<IREE::IO::Stream::WriteBytesOp>>(
      context, importSymbols, typeConverter, "io_stream.write.bytes");
  patterns.insert<ExpandWriteBytesOpConversion>(context);
  patterns.insert<WriteLineOpConversion>(context);
}

} // namespace mlir::iree_compiler
