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

#include <string>

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class RemoveExecutableOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ExecutableOp op, ArrayRef<Value> operands,
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

  LogicalResult matchAndRewrite(
      IREE::HAL::ExecutableCreateOp createOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = createOp.getLoc();
    IREE::HAL::ExecutableCreateOp::Adaptor newOperands(operands);

    auto funcOp = dyn_cast_or_null<IREE::VM::FuncOp>(
        rewriter.getInsertionBlock()->getParentOp());
    assert(funcOp && "prepare op not in a function");

    // Materialize vm.rodata for the binary.
    auto executableBinaryOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableBinaryOp>(
            createOp, createOp.executable_target());
    auto executableOp = executableBinaryOp.getOperation()
                            ->getParentOfType<IREE::HAL::ExecutableOp>();
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(funcOp);
    auto rodataOp = rewriter.create<IREE::VM::RodataOp>(
        executableBinaryOp.getLoc(),
        (StringRef("_") + executableOp.getName() + "_" +
         executableBinaryOp.getName() + "_binary_" +
         executableBinaryOp.format().lower())
            .str(),
        executableBinaryOp.data());
    rodataOp.setPrivate();
    if (executableBinaryOp.mime_type().hasValue()) {
      rodataOp.mime_typeAttr(executableBinaryOp.mime_typeAttr());
    }
    rewriter.restoreInsertionPoint(insertPoint);

    auto executableFormatString = detail::rewriteAttrToOperands(
        createOp.getLoc(), executableBinaryOp.formatAttr(),
        importOp.getType().getInput(1), rewriter);
    assert(executableFormatString.hasValue() &&
           executableFormatString.getValue().size() == 1);

    SmallVector<int16_t, 4> segmentSizes = {
        /*device=*/-1,
        /*executable_format=*/-1,
        /*executable_data=*/-1,
        /*executable_layouts=*/
        static_cast<int16_t>(llvm::size(newOperands.layouts())),
    };
    SmallVector<Value, 8> callOperands = {
        newOperands.device(),
        executableFormatString.getValue().front(),
        rewriter.createOrFold<IREE::VM::ConstRefRodataOp>(loc, rodataOp),
    };
    callOperands.append(newOperands.layouts().begin(),
                        newOperands.layouts().end());

    auto importType = importOp.getType();
    rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        createOp, rewriter.getSymbolRefAttr(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);

    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};  // namespace

}  // namespace

void populateHALExecutableToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       OwningRewritePatternList &patterns) {
  // hal.executables are not needed after conversion as we extract their
  // contents during conversion of the ops that use them.
  patterns.insert<RemoveExecutableOpConversion>(context);

  patterns.insert<ExecutableCreateOpConversion>(
      context, importSymbols, typeConverter, "hal.executable.create");

  patterns.insert<VMImportOpConversion<IREE::HAL::DescriptorSetLayoutCreateOp>>(
      context, importSymbols, typeConverter,
      "hal.descriptor_set_layout.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExecutableLayoutCreateOp>>(
      context, importSymbols, typeConverter, "hal.executable_layout.create");
}

}  // namespace iree_compiler
}  // namespace mlir
