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
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Convert operations which don't have attributes
template <typename SrcOpTy>
class NoAttributeOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  NoAttributeOpConversion(MLIRContext *context, StringRef funcName)
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
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  ShiftArithmeticOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(
        operands, op.getOperation()->getAttrDictionary());

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIndexType(), 0), srcAdaptor.amount()});
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getType(), callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

// TODO(simon-camp): These conversions to macro calls should be deleted once
// support for control flow ops has landed in the c module target
template <typename SrcOpTy>
class BinaryCheckOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  BinaryCheckOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(
        operands, op.getOperation()->getAttrDictionary());

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIndexType(), 0),
         IntegerAttr::get(rewriter.getIndexType(), 1), srcAdaptor.message()});
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, mlir::TypeRange{}, callee,
                                               args, templateArgs, operands);

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
  // Native integer arithmetic ops
  patterns.insert<NoAttributeOpConversion<IREE::VM::AddI32Op>>(context,
                                                               "vm_add_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::SubI32Op>>(context,
                                                               "vm_sub_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::MulI32Op>>(context,
                                                               "vm_mul_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::DivI32SOp>>(context,
                                                                "vm_div_i32s");
  patterns.insert<NoAttributeOpConversion<IREE::VM::DivI32UOp>>(context,
                                                                "vm_div_i32u");
  patterns.insert<NoAttributeOpConversion<IREE::VM::RemI32SOp>>(context,
                                                                "vm_rem_i32s");
  patterns.insert<NoAttributeOpConversion<IREE::VM::RemI32UOp>>(context,
                                                                "vm_rem_i32u");
  patterns.insert<NoAttributeOpConversion<IREE::VM::NotI32Op>>(context,
                                                               "vm_not_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::AndI32Op>>(context,
                                                               "vm_and_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::OrI32Op>>(context,
                                                              "vm_or_i32");
  patterns.insert<NoAttributeOpConversion<IREE::VM::XorI32Op>>(context,
                                                               "vm_xor_i32");

  // Native bitwise shift and rotate ops
  patterns.insert<ShiftArithmeticOpConversion<IREE::VM::ShlI32Op>>(
      context, "vm_shl_i32");
  patterns.insert<ShiftArithmeticOpConversion<IREE::VM::ShrI32SOp>>(
      context, "vm_shr_i32s");
  patterns.insert<ShiftArithmeticOpConversion<IREE::VM::ShrI32UOp>>(
      context, "vm_shr_i32u");

  // Check
  // TODO(simon-camp): These conversions to macro calls should be deleted once
  // support for control flow ops has landed in the c module target
  patterns.insert<BinaryCheckOpConversion<IREE::VM::CheckEQOp>>(context,
                                                                "VM_CHECK_EQ");

  // Compare
  patterns.insert<NoAttributeOpConversion<IREE::VM::CmpNEI32Op>>(
      context, "vm_cmp_ne_i32");

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
    registry.insert<mlir::emitc::EmitCDialect, IREEDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns;
    populateVMToCPatterns(&getContext(), patterns);

    target.addLegalDialect<mlir::emitc::EmitCDialect>();
    target.addLegalDialect<iree_compiler::IREEDialect>();
    target.addLegalDialect<IREE::VM::VMDialect>();

    // Native integer arithmetic ops
    target.addIllegalOp<IREE::VM::AddI32Op>();
    target.addIllegalOp<IREE::VM::SubI32Op>();
    target.addIllegalOp<IREE::VM::MulI32Op>();
    target.addIllegalOp<IREE::VM::DivI32SOp>();
    target.addIllegalOp<IREE::VM::DivI32UOp>();
    target.addIllegalOp<IREE::VM::RemI32SOp>();
    target.addIllegalOp<IREE::VM::RemI32UOp>();
    target.addIllegalOp<IREE::VM::NotI32Op>();
    target.addIllegalOp<IREE::VM::AndI32Op>();
    target.addIllegalOp<IREE::VM::OrI32Op>();
    target.addIllegalOp<IREE::VM::XorI32Op>();

    // Native bitwise shift and rotate ops
    target.addIllegalOp<IREE::VM::ShlI32Op>();
    target.addIllegalOp<IREE::VM::ShrI32SOp>();
    target.addIllegalOp<IREE::VM::ShrI32UOp>();

    // Check ops
    // TODO(simon-camp): These conversions to macro calls should be deleted once
    // support for control flow ops has landed in the c module target
    target.addIllegalOp<IREE::VM::CheckEQOp>();

    // Compare ops
    target.addIllegalOp<IREE::VM::CmpNEI32Op>();

    // Const ops
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
