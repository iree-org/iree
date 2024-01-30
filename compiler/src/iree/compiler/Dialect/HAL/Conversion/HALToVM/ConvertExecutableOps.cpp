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

IREE::VM::RodataOp
createExecutableBinaryRodata(IREE::HAL::ExecutableBinaryOp binaryOp,
                             OpBuilder &builder) {
  auto executableOp =
      binaryOp.getOperation()->getParentOfType<IREE::HAL::ExecutableOp>();
  auto insertPoint = builder.saveInsertionPoint();
  builder.setInsertionPoint(builder.getInsertionBlock()->getParentOp());

  std::string rodataName = sanitizeSymbolName(
      (executableOp.getName() + "_" + binaryOp.getName()).str());
  auto rodataOp = builder.create<IREE::VM::RodataOp>(
      binaryOp.getLoc(), rodataName, binaryOp.getData());
  rodataOp.setPrivate();
  if (binaryOp.getMimeType().has_value()) {
    rodataOp.setMimeTypeAttr(binaryOp.getMimeTypeAttr());
  }

  // TODO(benvanik): should these be page aligned? memcpy fastpath is fine for
  // now.
  rodataOp.setAlignmentAttr(builder.getI64IntegerAttr(16));

  builder.restoreInsertionPoint(insertPoint);

  return rodataOp;
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
    auto rodataOp = createExecutableBinaryRodata(executableBinaryOp, rewriter);
    auto executableRodata = rewriter.createOrFold<IREE::VM::ConstRefRodataOp>(
        createOp.getLoc(), rodataOp);

    // Get format string as a rodata blob.
    auto executableFormatStr = rewriter.create<IREE::VM::RodataInlineOp>(
        createOp.getLoc(), executableBinaryOp.getFormatAttr());

    // Pack constants, if any.
    auto constantBuffer = createPackedConstantBuffer(
        createOp.getLoc(), adaptor.getConstants(), rewriter);

    SmallVector<int16_t, 5> segmentSizes = {
        /*device=*/-1,
        /*executable_format=*/-1,
        /*executable_data=*/-1,
        /*constants=*/-1,
        /*pipeline_layouts=*/
        static_cast<int16_t>(llvm::size(adaptor.getLayouts())),
    };
    SmallVector<Value, 8> callOperands = {
        adaptor.getDevice(),
        executableFormatStr,
        executableRodata,
        constantBuffer,
    };
    callOperands.append(adaptor.getLayouts().begin(),
                        adaptor.getLayouts().end());

    auto importType = importOp.getFunctionType();
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        createOp, SymbolRefAttr::get(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
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

  patterns.insert<VMImportOpConversion<IREE::HAL::DescriptorSetLayoutCreateOp>>(
      context, importSymbols, typeConverter,
      "hal.descriptor_set_layout.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::PipelineLayoutCreateOp>>(
      context, importSymbols, typeConverter, "hal.pipeline_layout.create");
}

} // namespace mlir::iree_compiler
