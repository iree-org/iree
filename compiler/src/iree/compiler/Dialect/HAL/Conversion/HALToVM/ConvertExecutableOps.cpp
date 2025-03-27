// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Creates a !vm.buffer containing all of the |constantValues|.
// TODO(benvanik): if there are a decent number of actual constant values we
// should create a rodata buffer that we can clone and then poke in the
// dynamic values; currently we require a lot of IR in order to store each
// value and it's very wasteful (think potentially KB of binary size) in order
// to do this all dynamically.
Value createPackedConstantBuffer(Location loc, ValueRange constantValues,
                                 OpBuilder &builder) {
  auto bufferRefType =
      IREE::VM::RefType::get(builder.getType<IREE::VM::BufferType>());
  size_t constantCount = constantValues.size();
  if (constantValues.empty()) {
    // No constants; pass a null buffer.
    return builder.create<IREE::VM::ConstRefZeroOp>(loc, bufferRefType);
  }

  // Create the constant storage buffer.
  SmallVector<Location> constantLocs;
  constantLocs.reserve(constantCount);
  for (auto constantValue : constantValues) {
    constantLocs.push_back(constantValue.getLoc());
  }
  auto constantBufferLoc = builder.getFusedLoc(constantLocs);
  auto constantBuffer = builder.create<IREE::VM::BufferAllocOp>(
      constantBufferLoc, bufferRefType,
      builder.create<IREE::VM::ConstI64Op>(constantBufferLoc,
                                           constantCount * sizeof(uint32_t)),
      builder.create<IREE::VM::ConstI32Op>(constantBufferLoc, 16));

  // Store each constant into it.
  // TODO(#8477): better ops for this pattern; this creates a lot of
  // extra IR for the indices. We should batch them up and append in one go.
  for (auto constantValue : llvm::enumerate(constantValues)) {
    // Buffer is zero-initialized so we can skip zero values.
    if (mlir::matchPattern(constantValue.value(), m_Zero()))
      continue;
    auto constantLoc = constantValue.value().getLoc();
    builder.create<IREE::VM::BufferStoreI32Op>(
        constantLoc, constantBuffer,
        builder.create<IREE::VM::ConstI64Op>(constantLoc,
                                             constantValue.index()),
        constantValue.value());
  }

  return constantBuffer;
}

namespace {

class RemoveExecutableOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ExecutableCreateOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableCreateOp> {
public:
  ExecutableCreateOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                               TypeConverter &typeConverter,
                               StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableCreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Materialize vm.rodata for the binary.
    auto executableBinaryOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableBinaryOp>(
            createOp, createOp.getExecutableTarget());
    auto executableOp = executableBinaryOp.getOperation()
                            ->getParentOfType<IREE::HAL::ExecutableOp>();
    std::string rodataName = sanitizeSymbolName(
        (executableOp.getName() + "_" + executableBinaryOp.getName()).str());
    auto rodataOp = rewriter.create<IREE::VM::RodataInlineOp>(
        executableBinaryOp.getLoc(),
        IREE::VM::RefType::get(rewriter.getType<IREE::VM::BufferType>()),
        rewriter.getStringAttr(rodataName), executableBinaryOp.getData(),
        rewriter.getI64IntegerAttr(16), executableBinaryOp.getMimeTypeAttr());

    // Get format string as a rodata blob.
    auto executableFormatStr = rewriter.create<IREE::VM::RodataInlineOp>(
        createOp.getLoc(), executableBinaryOp.getFormatAttr());

    // Pack constants, if any.
    auto constantBuffer = createPackedConstantBuffer(
        createOp.getLoc(), adaptor.getConstants(), rewriter);

    SmallVector<Value, 8> callOperands = {
        adaptor.getDevice(), adaptor.getQueueAffinity(),
        executableFormatStr, rodataOp,
        constantBuffer,
    };
    auto importType = importOp.getFunctionType();
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        createOp, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);
    copyImportAttrs(importOp, callOp);

    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

} // namespace

void populateHALExecutableToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // hal.executables are not needed after conversion as we extract their
  // contents during conversion of the ops that use them.
  patterns.insert<RemoveExecutableOpConversion>(context);

  patterns.insert<ExecutableCreateOpConversion>(
      context, importSymbols, typeConverter, "hal.executable.create");
}

} // namespace mlir::iree_compiler
