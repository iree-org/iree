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
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
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

class ExecutableCachePrepareOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableCachePrepareOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ExecutableCachePrepareOp prepareOp,
      llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = prepareOp.getLoc();
    IREE::HAL::ExecutableCachePrepareOp::Adaptor newOperands(operands);

    auto funcOp = dyn_cast_or_null<IREE::VM::FuncOp>(
        rewriter.getInsertionBlock()->getParentOp());
    assert(funcOp && "prepare op not in a function");

    // Materialize vm.rodata for each binary format available.
    SmallVector<Attribute, 4> availableFormatAttrs;
    SmallVector<IREE::VM::RodataOp, 4> rodataOps;
    auto executableOp =
        cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
            prepareOp, prepareOp.executable()));
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(funcOp);
    for (auto binaryOp :
         executableOp.getBlock().getOps<IREE::HAL::ExecutableBinaryOp>()) {
      availableFormatAttrs.push_back(binaryOp.formatAttr());
      auto rodataOp = rewriter.create<IREE::VM::RodataOp>(
          binaryOp.getLoc(),
          (StringRef("_") + executableOp.getName() + "_binary_" +
           IREE::HAL::stringifyExecutableFormat(binaryOp.format()).lower())
              .str(),
          binaryOp.data());
      SymbolTable::setSymbolVisibility(rodataOp,
                                       SymbolTable::Visibility::Private);
      rodataOps.push_back(rodataOp);
    }
    rewriter.restoreInsertionPoint(insertPoint);

    // Get the index of the format the cache prefers. Returns -1 if none
    // available which will cause the fallthrough on the select.
    auto indexValue =
        rewriter.createOrFold<IREE::HAL::ExecutableCacheSelectFormatOp>(
            loc, rewriter.getIntegerType(32), newOperands.executable_cache(),
            rewriter.getArrayAttr(availableFormatAttrs));

    // Select the byte buffer based on the preferred format.
    SmallVector<Value, 4> rodataValues;
    for (auto rodataOp : rodataOps) {
      rodataValues.push_back(
          rewriter.createOrFold<IREE::VM::ConstRefRodataOp>(loc, rodataOp));
    }
    auto byteBufferRefType = IREE::VM::RefType::get(
        IREE::ByteBufferType::get(rewriter.getContext()));
    auto defaultValue =
        rewriter.createOrFold<IREE::VM::ConstRefZeroOp>(loc, byteBufferRefType);
    auto chosenByteBufferValue = rewriter.createOrFold<IREE::VM::SwitchRefOp>(
        loc, byteBufferRefType, indexValue, defaultValue, rodataValues);

    // Call the import method with the byte buffer. Note that if no format
    // was available then we will default to null and preparation will fail.
    // We could instead check here in the IR and change our behavior, but the
    // error message from the failed prepare is good enough.
    auto executableRefType = IREE::VM::RefType::get(
        IREE::HAL::ExecutableType::get(rewriter.getContext()));
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        prepareOp, "hal.executable_cache.prepare",
        ArrayRef<Type>{executableRefType},
        ArrayRef<Value>{newOperands.executable_cache(),
                        newOperands.executable_layout(),
                        rewriter.createOrFold<IREE::VM::ConstI32Op>(
                            loc, prepareOp.caching_modeAttr()),
                        chosenByteBufferValue});

    return success();
  }
};

}  // namespace

void populateHALExecutableToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       OwningRewritePatternList &patterns) {
  // hal.executables are not needed after conversion as we extract their
  // contents during conversion of the ops that use them.
  patterns.insert<RemoveExecutableOpConversion>(context);

  patterns.insert<VMImportOpConversion<IREE::HAL::ExecutableCacheCreateOp>>(
      context, importSymbols, typeConverter, "hal.executable_cache.create");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::ExecutableCacheSelectFormatOp>>(
          context, importSymbols, typeConverter,
          "hal.executable_cache.select_format");
  patterns.insert<ExecutableCachePrepareOpConversion>(context);

  patterns.insert<VMImportOpConversion<IREE::HAL::DescriptorSetLayoutCreateOp>>(
      context, importSymbols, typeConverter,
      "hal.descriptor_set_layout.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExecutableLayoutCreateOp>>(
      context, importSymbols, typeConverter, "hal.executable_layout.create");
}

}  // namespace iree_compiler
}  // namespace mlir
