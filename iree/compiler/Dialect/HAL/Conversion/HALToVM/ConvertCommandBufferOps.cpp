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
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class CommandBufferPushDescriptorSetOpConversion
    : public OpConversionPattern<IREE::HAL::CommandBufferPushDescriptorSetOp> {
 public:
  CommandBufferPushDescriptorSetOpConversion(MLIRContext *context,
                                             SymbolTable &importSymbols,
                                             TypeConverter &typeConverter,
                                             StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::CommandBufferPushDescriptorSetOp op,
      llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();
    IREE::HAL::CommandBufferPushDescriptorSetOp::Adaptor newOperands(operands);

    SmallVector<Value, 8> callOperands = {
        newOperands.command_buffer(),
        newOperands.executable_layout(),
        rewriter.create<mlir::ConstantOp>(
            op.getLoc(), rewriter.getI32IntegerAttr(
                             static_cast<int32_t>(op.setAttr().getInt()))),
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*command_buffer=*/-1,
        /*executable_layout=*/-1,
        /*set=*/-1,
        /*bindings_ordinals=*/
        static_cast<int16_t>(op.bindings().size()),
        /*bindings_buffers=*/
        static_cast<int16_t>(op.bindings().size()),
        /*bindings_offsets=*/
        static_cast<int16_t>(op.bindings().size()),
        /*bindings_lengths=*/
        static_cast<int16_t>(op.bindings().size()),
    };
    for (auto bindingAttr : op.bindings()) {
      callOperands.push_back(
          rewriter.create<mlir::ConstantOp>(op.getLoc(), bindingAttr));
    }
    for (auto bindingBuffer : newOperands.binding_buffers()) {
      callOperands.push_back(bindingBuffer);
    }
    for (auto bindingOffset : newOperands.binding_offsets()) {
      callOperands.push_back(bindingOffset);
    }
    for (auto bindingLength : newOperands.binding_lengths()) {
      callOperands.push_back(bindingLength);
    }

    rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALCommandBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCreateOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferBeginOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.begin");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferEndOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.end");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferExecutionBarrierOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.execution_barrier");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferFillBufferOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.fill_buffer");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferCopyBufferOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.copy_buffer");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferPushConstantsOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.push_constants");
  patterns.insert<CommandBufferPushDescriptorSetOpConversion>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.push_descriptor_set");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::CommandBufferBindDescriptorSetOp>>(
      context, importSymbols, typeConverter,
      "hal.command_buffer.bind_descriptor_set");
  patterns.insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchOp>>(
      context, importSymbols, typeConverter, "hal.command_buffer.dispatch");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::CommandBufferDispatchIndirectOp>>(
          context, importSymbols, typeConverter,
          "hal.command_buffer.dispatch.indirect");
}

}  // namespace iree_compiler
}  // namespace mlir
