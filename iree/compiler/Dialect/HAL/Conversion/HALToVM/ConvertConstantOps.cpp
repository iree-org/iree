// Copyright 2020 Google LLC
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
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConstantPoolOpConversion
    : public OpConversionPattern<IREE::HAL::ConstantPoolOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantPoolOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    for (auto storageOp : op.getOps<IREE::HAL::ConstantStorageOp>()) {
      auto rodataName = (op.sym_name() + storageOp.sym_name()).str();
      auto rodataOp = rewriter.create<IREE::VM::RodataOp>(
          storageOp.getLoc(), rodataName, storageOp.value());
      SymbolTable::setSymbolVisibility(rodataOp,
                                       SymbolTable::Visibility::Private);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class ConstantStorageLookupOpConversion
    : public OpConversionPattern<IREE::HAL::ConstantStorageLookupOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantStorageLookupOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // I don't like this, but I can't figure out what to do.
    // Matches the logic above.
    auto rodataName =
        (op.constant().getRootReference() + op.constant().getLeafReference())
            .str();
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefRodataOp>(op, rodataName);
    return success();
  }
};

}  // namespace

void populateHALConstantToVMPatterns(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<ConstantPoolOpConversion, ConstantStorageLookupOpConversion>(
      typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
