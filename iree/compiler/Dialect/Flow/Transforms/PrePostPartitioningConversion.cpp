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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
namespace {
/// ExtractElementOp will be lowered to IREE::Flow::TensorLoadOp. If the type is
/// i1, it's not valid to load. In this case, we need to cast it to i8 before
/// the load, and truncate the value after the load.
struct ExtractElementOpPromotion
    : public OpConversionPattern<ExtractElementOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ExtractElementOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    auto tensorType = op.getAggregate().getType().dyn_cast<TensorType>();
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "expected tensor types");
    }
    if (tensorType.getElementTypeBitWidth() != 1) {
      return rewriter.notifyMatchFailure(op, "expected i1 type");
    }
    Location loc = op.getLoc();
    auto i8Type = rewriter.getIntegerType(8);
    auto i8Operand = rewriter.create<mhlo::ConvertOp>(loc, args[0], i8Type);
    auto loadOp =
        rewriter.create<ExtractElementOp>(loc, i8Type, i8Operand, op.indices());
    auto i1Type = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<TruncateIOp>(op, i1Type, loadOp.getResult());
    return success();
  }
};
}  // namespace

class PrePartitioningConversionPass
    : public PassWrapper<PrePartitioningConversionPass, FunctionPass> {
 public:
  void runOnFunction() override {
    auto *context = &getContext();
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
    conversionTarget.addLegalDialect<mhlo::MhloDialect>();

    // Control flow must be converted to standard form via
    // mhlo::createLegalizeControlFlowPass() prior to conversion.
    conversionTarget.addIllegalOp<mhlo::IfOp, mhlo::CaseOp, mhlo::WhileOp>();

    // We don't support broadcast_dimensions as part of ops, so materialize
    // any such attributes to dedicated mhlo.broadcast_in_dim ops.
    mhlo::SetupMaterializeBroadcastsLegality(context, &conversionTarget);
    mhlo::PopulateMaterializeBroadcastsPatterns(context, &conversionPatterns);

    // Early conversion of ops that have matches we want to route through.
    // For example, DynamicUpdateSlice should end up as a stream operation.
    setupDirectHLOToFlowLegality(context, conversionTarget);
    populateHLOToFlowPatterns(context, conversionPatterns);
    setupDirectStandardToFlowLegality(context, conversionTarget);
    conversionPatterns.insert<ExtractElementOpPromotion>(context);
    populateStandardToFlowPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      getFunction().emitError() << "module is not in a compatible input format";
      return signalPassFailure();
    }

    for (Operation &op : getFunction().getOps()) {
      if (!FlowDialect::isDialectOp(&op)) continue;
      bool hasI1Type = false;
      for (auto type : op.getOperandTypes()) {
        if (type.isSignlessInteger() && type.getIntOrFloatBitWidth() == 1) {
          hasI1Type = true;
          break;
        }
        auto shapedType = type.dyn_cast<ShapedType>();
        if (shapedType && shapedType.getElementTypeBitWidth() == 1) {
          hasI1Type = true;
          break;
        }
      }
      if (hasI1Type) {
        getFunction().emitError() << "expected non-i1 types in FlowDialect";
        return signalPassFailure();
      }
    }
  }
};

class PostPartitioningConversionPass
    : public PassWrapper<PostPartitioningConversionPass, FunctionPass> {
 public:
  void runOnFunction() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(getContext());
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
                                      conversionPatterns))) {
      getFunction().emitError() << "module is not in a compatible input format";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createPrePartitioningConversionPass() {
  return std::make_unique<PrePartitioningConversionPass>();
}

std::unique_ptr<OperationPass<FuncOp>> createPostPartitioningConversionPass() {
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
