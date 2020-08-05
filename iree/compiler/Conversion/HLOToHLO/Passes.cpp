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

#include "iree/compiler/Conversion/HLOToHLO/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct ConvertHLOToCompatibleHLOPass
    : public PassWrapper<ConvertHLOToCompatibleHLOPass, FunctionPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList greedyPatterns;
    mhlo::PopulateComplexLoweringPatterns(context, &greedyPatterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), greedyPatterns))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createHLOToCompatibleHLOPass() {
  return std::make_unique<ConvertHLOToCompatibleHLOPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
