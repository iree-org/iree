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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstOpLowering : public OpRewritePattern<xla_hlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(xla_hlo::ConstOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.value());
    return matchSuccess();
  }
};

// TODO(benvanik): dynamic update slice.

}  // namespace

void setupHLOToFlowConversion(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              OwningRewritePatternList &patterns) {
  conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();

  // Standard ops always pass through as import code may have produced some
  // and control flow should have been legalized from HLO to std.
  // The flow dialect uses std.module and std.func for its structure and they
  // must be allowed.
  conversionTarget.addLegalDialect<StandardOpsDialect>();
  conversionTarget.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();

  // Allow all non-blacklisted HLO ops by default. Partitioning will move most
  // of them into executables.
  conversionTarget.addLegalDialect<xla_hlo::XlaHloDialect>();

  // Control flow must be converted to standard form via
  // xla_hlo::createLegalizeControlFlowPass() prior to conversion.
  conversionTarget.addIllegalOp<xla_hlo::ConditionalOp, xla_hlo::WhileOp>();

  conversionTarget.addIllegalOp<xla_hlo::DotGeneralOp>();
  xla_hlo::PopulateGeneralDotOpLoweringPatterns(&patterns, context);

  conversionTarget.addIllegalOp<xla_hlo::ConstOp>();
  patterns.insert<ConstOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
