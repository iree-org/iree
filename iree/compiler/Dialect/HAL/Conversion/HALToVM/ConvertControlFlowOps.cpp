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
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class CheckSuccessOpConversion
    : public OpConversionPattern<IREE::HAL::CheckSuccessOp> {
 public:
  CheckSuccessOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                           TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::CheckSuccessOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If status value is non-zero, fail.
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(
        op, op.status(), op.message().getValueOr(""));
    return success();
  }
};

void populateHALControlFlowToVMPatterns(MLIRContext *context,
                                        SymbolTable &importSymbols,
                                        TypeConverter &typeConverter,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<CheckSuccessOpConversion>(context, importSymbols,
                                            typeConverter, "hal.check_success");
}

}  // namespace iree_compiler
}  // namespace mlir
