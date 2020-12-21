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

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Convert operations which only have operands
template <typename SrcOpTy>
class OperandOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  OperandOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getType(), callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

template <typename SrcOpTy>
class ConstOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  ConstOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args = ArrayAttr::get({op.value()}, op.getContext());
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getType(), callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

}  // namespace

void populateVMToCPatterns(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  // Arithmetic
  patterns.insert<OperandOpConversion<IREE::VM::AddI32Op>>(context,
                                                           "vm_add_i32");

  // Compare
  patterns.insert<OperandOpConversion<IREE::VM::CmpNEI32Op>>(context,
                                                             "vm_cmp_ne_i32");

  // Const
  patterns.insert<ConstOpConversion<IREE::VM::ConstI32Op>>(context,
                                                           "vm_const_i32");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns;
    populateVMToCPatterns(&getContext(), patterns);

    target.addLegalDialect<mlir::emitc::EmitCDialect>();
    target.addLegalDialect<IREE::VM::VMDialect>();

    // Arithmetic
    target.addIllegalOp<IREE::VM::AddI32Op>();

    // Compare
    target.addIllegalOp<IREE::VM::CmpNEI32Op>();

    // Const
    target.addIllegalOp<IREE::VM::ConstI32Op>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}

}  // namespace VM
}  // namespace IREE

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass(
    "iree-convert-vm-to-emitc", "Convert VM Ops to the EmitC dialect");

}  // namespace iree_compiler
}  // namespace mlir
