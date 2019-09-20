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

#include "iree/compiler/IR/Dialect.h"
#include "iree/compiler/IR/Interpreter/HLDialect.h"
#include "iree/compiler/IR/Interpreter/LLDialect.h"
#include "iree/compiler/Transforms/Interpreter/Rewrites.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/StandardOps/Ops.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/PatternMatch.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/DialectConversion.h"
#include "third_party/tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

class LowerToInterpreterDialectPass
    : public FunctionPass<LowerToInterpreterDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto* ctx = &getContext();
    xla_hlo::PopulateGeneralDotOpLoweringPatterns(&patterns, ctx);
    populateLowerStdToInterpreterPatterns(patterns, ctx);
    xla_hlo::PopulateXlaToStdPatterns(&patterns, ctx);
    populateLowerXlaToInterpreterPatterns(patterns, ctx);

    ConversionTarget target(getContext());
    target.addLegalDialect<IREEHLInterpreterDialect,
                           // TODO(b/139012931) Reduce lowerings create LL ops
                           // for some reason
                           IREELLInterpreterDialect, IREEDialect>();
    target.addLegalOp<LoadOp, StoreOp, FuncOp, ReturnOp>();
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp constOp) {
      // std.constant is legal for index integers.
      return constOp.getValue().isa<IntegerAttr>() &&
             constOp.getType().isIndex();
    });
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLowerToInterpreterDialectPass() {
  return std::make_unique<LowerToInterpreterDialectPass>();
}

static PassRegistration<LowerToInterpreterDialectPass> pass(
    "lower-to-iree-interpreter",
    "Convert all ops to the IREE interpreter dialect");

}  // namespace iree_compiler
}  // namespace mlir
