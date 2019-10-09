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
#include "iree/compiler/IR/Sequencer/HLDialect.h"
#include "iree/compiler/IR/Sequencer/LLDialect.h"
#include "iree/compiler/Transforms/Rewrites.h"
#include "iree/compiler/Transforms/Sequencer/Rewrites.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

class LowerToSequencerDialectPass
    : public FunctionPass<LowerToSequencerDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto* ctx = &getContext();
    MemRefTypeConverter converter(ctx);
    xla_hlo::PopulateGeneralDotOpLoweringPatterns(&patterns, ctx);
    xla_hlo::PopulateXlaToStdPatterns(&patterns, ctx);
    populateLowerStdToIreePatterns(patterns, ctx);
    populateLowerStdToSequencerPatterns(patterns, converter, ctx);
    populateLowerXlaToIreePatterns(patterns, ctx);
    populateLowerXlaToSequencerPatterns(patterns, ctx);

    ConversionTarget target(getContext());
    target.addLegalDialect<IREEHLSequencerDialect, IREEDialect>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLowerToSequencerDialectPass() {
  return std::make_unique<LowerToSequencerDialectPass>();
}

static PassRegistration<LowerToSequencerDialectPass> pass(
    "lower-to-iree-sequencer", "Convert all ops to the IREE sequencer dialect");

}  // namespace iree_compiler
}  // namespace mlir
