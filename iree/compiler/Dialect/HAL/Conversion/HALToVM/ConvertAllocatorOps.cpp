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
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class AllocatorMapOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorMapOp> {
 public:
  AllocatorMapOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    wrapByteBufferImportOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "hal.allocator.wrap.byte_buffer");
    assert(wrapByteBufferImportOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::AllocatorMapOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::AllocatorMapOp::Adaptor opAdaptor(operands);
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, wrapByteBufferImportOp.getName(),
        ArrayRef<Type>{getTypeConverter()->convertType(op.getType())},
        ArrayRef<Value>{opAdaptor.allocator(),
                        rewriter.createOrFold<IREE::VM::ConstI32Op>(
                            op.getLoc(), op.memory_typesAttr()),
                        rewriter.createOrFold<IREE::VM::ConstI32Op>(
                            op.getLoc(), op.buffer_usageAttr()),
                        opAdaptor.source(), opAdaptor.offset(),
                        opAdaptor.length()});
    return success();
  }

 private:
  mutable IREE::VM::ImportOp wrapByteBufferImportOp;
};

}  // namespace

void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorAllocateOp>>(
      context, importSymbols, typeConverter, "hal.allocator.allocate");
  patterns.insert<AllocatorMapOpConversion>(typeConverter, context,
                                            importSymbols);
}

}  // namespace iree_compiler
}  // namespace mlir
