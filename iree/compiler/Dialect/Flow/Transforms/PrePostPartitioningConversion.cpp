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

#include <utility>

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"
#include "iree/compiler/Dialect/Flow/Conversion/StandardToFlow/ConvertStandardToFlow.h"
#include "iree/compiler/Dialect/Flow/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class PrePartitioningConversionPass
    : public FunctionPass<PrePartitioningConversionPass> {
 public:
  void runOnFunction() override {
    auto *context = &getContext();
    FlowTypeConverter typeConverter;
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns;

    conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();

    // Standard ops always pass through as import code may have produced some
    // and control flow should have been legalized from HLO to std.
    // The flow dialect uses std.module and std.func for its structure and they
    // must be allowed.
    conversionTarget.addLegalDialect<StandardOpsDialect>();
    conversionTarget.addLegalOp<FuncOp>();

    // Allow XLA HLO ops - we explicitly mark the ones we don't want below.
    conversionTarget.addLegalDialect<xla_hlo::XlaHloDialect>();

    // Control flow must be converted to standard form via
    // xla_hlo::createLegalizeControlFlowPass() prior to conversion.
    conversionTarget.addIllegalOp<xla_hlo::ConditionalOp, xla_hlo::WhileOp>();

    conversionTarget.addIllegalOp<xla_hlo::DotGeneralOp>();
    xla_hlo::PopulateGeneralDotOpLoweringPatterns(&conversionPatterns, context);

    // We don't support broadcast_dimensions as part of ops, so materialize
    // any such attributes to dedicated xla_hlo.broadcast_in_dim ops.
    xla_hlo::SetupMaterializeBroadcastsLegality(context, &conversionTarget);
    xla_hlo::PopulateMaterializeBroadcastsPatterns(context,
                                                   &conversionPatterns);

    // Early conversion of ops that have matches we want to route through.
    // For example, DynamicUpdateSlice should end up as a stream operation.
    setupDirectHLOToFlowLegality(context, conversionTarget);
    populateHLOToFlowPatterns(context, conversionPatterns);
    setupDirectStandardToFlowLegality(context, conversionTarget);
    populateStandardToFlowPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns, &typeConverter))) {
      getFunction().emitError() << "module is not in a compatible input format";
      return signalPassFailure();
    }
  }
};

class PostPartitioningConversionPass
    : public FunctionPass<PostPartitioningConversionPass> {
 public:
  void runOnFunction() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(getContext());
    FlowTypeConverter typeConverter;
    OwningRewritePatternList conversionPatterns;

    // We have completed all flow op creation at this point.
    conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();

    // Standard ops always pass through as import code may have produced some
    // and control flow should have been legalized from HLO to std.
    // The flow dialect uses std.module and std.func for its structure and they
    // must be allowed.
    conversionTarget.addLegalDialect<StandardOpsDialect>();
    conversionTarget.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();

    // Pick up any remaining HLO ops that were not partitioned.
    populateHLOToFlowPatterns(context, conversionPatterns);
    populateStandardToFlowPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns, &typeConverter))) {
      getFunction().emitError() << "module is not in a compatible input format";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createPrePartitioningConversionPass() {
  return std::make_unique<PrePartitioningConversionPass>();
}

std::unique_ptr<OpPassBase<FuncOp>> createPostPartitioningConversionPass() {
  return std::make_unique<PostPartitioningConversionPass>();
}

static PassRegistration<PrePartitioningConversionPass> prePass(
    "iree-flow-pre-partitioning-conversion",
    "Dialect conversion prior to partitioning");

static PassRegistration<PostPartitioningConversionPass> postPass(
    "iree-flow-post-partitioning-conversion",
    "Dialect conversion after partitioning");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
