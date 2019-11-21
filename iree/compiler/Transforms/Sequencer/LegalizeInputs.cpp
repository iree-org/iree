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
#include "iree/compiler/Transforms/Rewrites.h"
#include "iree/compiler/Transforms/Sequencer/Rewrites.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

class LegalizeInputOpsPass : public FunctionPass<LegalizeInputOpsPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    xla_hlo::PopulateGeneralDotOpLoweringPatterns(&patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<xla_hlo::XlaHloDialect, StandardOpsDialect>();
    target.addLegalOp<FuncOp, ReturnOp>();
    target.addIllegalOp<xla_hlo::DotGeneralOp, xla_hlo::WhileOp>();
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeInputOpsPass() {
  return std::make_unique<LegalizeInputOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
