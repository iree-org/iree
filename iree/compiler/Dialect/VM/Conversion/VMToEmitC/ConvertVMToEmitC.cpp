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

SmallVector<Attribute, 4> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

// Convert vm operations to emitc calls. The resultiong call has the ops
// operands as arguments followed by an argument for every attribute.
template <typename SrcOpTy>
class CallOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  CallOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute, 4> args_ =
        indexSequence(operands.size(), op.getContext());

    for (NamedAttribute attr : op.getAttrs()) {
      args_.push_back(attr.second);
    }

    auto type = op.getOperation()->getResultTypes();
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args = rewriter.getArrayAttr(args_);
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, type, callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

}  // namespace

void populateVMToCPatterns(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  // Native integer arithmetic ops
  patterns.insert<CallOpConversion<IREE::VM::AddI32Op>>(context, "vm_add_i32");
  patterns.insert<CallOpConversion<IREE::VM::SubI32Op>>(context, "vm_sub_i32");
  patterns.insert<CallOpConversion<IREE::VM::MulI32Op>>(context, "vm_mul_i32");
  patterns.insert<CallOpConversion<IREE::VM::DivI32SOp>>(context,
                                                         "vm_div_i32s");
  patterns.insert<CallOpConversion<IREE::VM::DivI32UOp>>(context,
                                                         "vm_div_i32u");
  patterns.insert<CallOpConversion<IREE::VM::RemI32SOp>>(context,
                                                         "vm_rem_i32s");
  patterns.insert<CallOpConversion<IREE::VM::RemI32UOp>>(context,
                                                         "vm_rem_i32u");
  patterns.insert<CallOpConversion<IREE::VM::NotI32Op>>(context, "vm_not_i32");
  patterns.insert<CallOpConversion<IREE::VM::AndI32Op>>(context, "vm_and_i32");
  patterns.insert<CallOpConversion<IREE::VM::OrI32Op>>(context, "vm_or_i32");
  patterns.insert<CallOpConversion<IREE::VM::XorI32Op>>(context, "vm_xor_i32");

  // Native bitwise shift and rotate ops
  patterns.insert<CallOpConversion<IREE::VM::ShlI32Op>>(context, "vm_shl_i32");
  patterns.insert<CallOpConversion<IREE::VM::ShrI32SOp>>(context,
                                                         "vm_shr_i32s");
  patterns.insert<CallOpConversion<IREE::VM::ShrI32UOp>>(context,
                                                         "vm_shr_i32u");

  // Comparison ops
  patterns.insert<CallOpConversion<IREE::VM::CmpEQI32Op>>(context,
                                                          "vm_cmp_eq_i32");
  patterns.insert<CallOpConversion<IREE::VM::CmpNEI32Op>>(context,
                                                          "vm_cmp_ne_i32");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI32SOp>>(context,
                                                           "vm_cmp_lt_i32s");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI32UOp>>(context,
                                                           "vm_cmp_lt_i32u");
  patterns.insert<CallOpConversion<IREE::VM::CmpNZI32Op>>(context,
                                                          "vm_cmp_nz_i32");

  // Check
  // TODO(simon-camp): These conversions to macro calls should be deleted once
  // support for control flow ops has landed in the c module target
  patterns.insert<CallOpConversion<IREE::VM::CheckEQOp>>(context,
                                                         "VM_CHECK_EQ");

  // Const
  patterns.insert<CallOpConversion<IREE::VM::ConstI32Op>>(context,
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

    // Comparison ops
    target.addIllegalOp<IREE::VM::CmpEQI32Op>();
    target.addIllegalOp<IREE::VM::CmpNEI32Op>();
    target.addIllegalOp<IREE::VM::CmpLTI32SOp>();
    target.addIllegalOp<IREE::VM::CmpLTI32UOp>();
    target.addIllegalOp<IREE::VM::CmpNZI32Op>();

    // Check ops
    // TODO(simon-camp): These conversions to macro calls should be deleted once
    // support for control flow ops has landed in the c module target
    target.addIllegalOp<IREE::VM::CheckEQOp>();

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
