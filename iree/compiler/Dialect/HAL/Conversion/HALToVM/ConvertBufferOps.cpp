// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class BufferLoadOpConversion
    : public OpConversionPattern<IREE::HAL::BufferLoadOp> {
 public:
  BufferLoadOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                         TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::BufferLoadOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::BufferLoadOp::Adaptor adaptor(operands);
    auto importType = importOp.getType();
    auto sizeConst = rewriter.createOrFold<mlir::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(
            IREE::HAL::getRoundedElementByteWidth(op.getResult().getType())));
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        ArrayRef<Value>{adaptor.source_buffer(), adaptor.source_offset(),
                        sizeConst});
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

class BufferStoreOpConversion
    : public OpConversionPattern<IREE::HAL::BufferStoreOp> {
 public:
  BufferStoreOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                          TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::BufferStoreOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::BufferStoreOp::Adaptor adaptor(operands);
    auto importType = importOp.getType();
    auto sizeConst = rewriter.createOrFold<mlir::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(
            IREE::HAL::getRoundedElementByteWidth(op.value().getType())));
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        ArrayRef<Value>{adaptor.value(), adaptor.target_buffer(),
                        adaptor.target_offset(), sizeConst});
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

void populateHALBufferToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAllocatorOp>>(
      context, importSymbols, typeConverter, "hal.buffer.allocator");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferSubspanOp>>(
      context, importSymbols, typeConverter, "hal.buffer.subspan");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferFillOp>>(
      context, importSymbols, typeConverter, "hal.buffer.fill");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferReadDataOp>>(
      context, importSymbols, typeConverter, "hal.buffer.read_data");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferWriteDataOp>>(
      context, importSymbols, typeConverter, "hal.buffer.write_data");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferCopyDataOp>>(
      context, importSymbols, typeConverter, "hal.buffer.copy_data");
  patterns.insert<BufferLoadOpConversion>(context, importSymbols, typeConverter,
                                          "hal.buffer.load");
  patterns.insert<BufferStoreOpConversion>(context, importSymbols,
                                           typeConverter, "hal.buffer.store");
}

}  // namespace iree_compiler
}  // namespace mlir
