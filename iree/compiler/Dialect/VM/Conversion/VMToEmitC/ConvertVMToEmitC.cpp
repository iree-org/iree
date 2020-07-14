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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Taken over from StandartToVM.
// We need to replace the Op depending on the operand.
// We could start with a conversion for IREE::VM::AddI32Op
template <typename SrcOpTy, typename DstOpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  BinaryArithmeticOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdapter(operands);

    MLIRContext *ctx = srcOp.getContext();

    // name of the function to call
    StringAttr callee_ = StringAttr::get(funcName, ctx);

    // attributes of the function call; references only the variable operands
    ArrayAttr args_ = ArrayAttr::get({IntegerAttr::get(IndexType::get(ctx), 0),
                                      IntegerAttr::get(IndexType::get(ctx), 1)},
                                     ctx);
    // operands of the function
    ValueRange operands_{srcAdapter.lhs(), srcAdapter.rhs()};

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcAdapter.lhs().getType(),
                                         callee_, args_, operands_);

    return success();
  }

  StringRef funcName;
};

}  // namespace

void populateVMToCPatterns(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  patterns.insert<
      BinaryArithmeticOpConversion<IREE::VM::AddI32Op, mlir::emitc::CallOp>>(
      context, "vm_add_i32");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::FuncOp>> {
  void runOnOperation() override {
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns;
    populateVMToCPatterns(&getContext(), patterns);

    target.addLegalDialect<mlir::emitc::EmitCDialect>();
    target.addLegalDialect<IREE::VM::VMDialect>();
    target.addIllegalOp<IREE::VM::AddI32Op>();

    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::FuncOp>> createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}
}  // namespace VM
}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir
