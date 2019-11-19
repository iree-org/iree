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

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// A pass converting XLA HLO operations into the IREE flow dialect.
// Used only for testing as in the common case we only rely on rewrite patterns.
class ConvertHLOToFlowPass : public ModulePass<ConvertHLOToFlowPass> {
  void runOnModule() override {
    OwningRewritePatternList patterns;
    auto module = getModule();

    ConversionTarget target(*module.getContext());
    setupHLOToFlowConversion(module.getContext(), target, patterns);

    // NOTE: we are only looking for specific HLO ops and allow others to
    // remain.
    if (failed(applyFullConversion(module, target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<ConvertHLOToFlowPass> pass(
    "iree-convert-hlo-to-flow", "Convert XLA HLO ops to the IREE flow dialect");

}  // namespace iree_compiler
}  // namespace mlir
