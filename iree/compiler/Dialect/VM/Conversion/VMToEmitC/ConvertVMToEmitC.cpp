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

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdapter(operands);

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcAdapter.lhs().getType(),
                                         srcAdapter.lhs(), srcAdapter.rhs());
    return success();
  }
};

}  // namespace

void populateVMToCPatterns(MLIRContext *context,
                           OwningRewritePatternList &patterns) {}

}  // namespace iree_compiler
}  // namespace mlir
