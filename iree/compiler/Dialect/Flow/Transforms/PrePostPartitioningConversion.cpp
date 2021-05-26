// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"
#include "iree/compiler/Dialect/Flow/Conversion/StandardToFlow/ConvertStandardToFlow.h"
#include "iree/compiler/Dialect/Flow/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Patterns.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class PrePartitioningConversionPass
    : public PrePartitioningConversionBase<PrePartitioningConversionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, FlowDialect, mhlo::MhloDialect,
                    StandardOpsDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns(&getContext());

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

    Shape::populateShapeToStandardConversionPatterns(conversionPatterns,
                                                     context);
    Shape::setupShapeToStandardLegality(conversionTarget);

    // Early conversion of ops that have matches we want to route through.
    // For example, DynamicUpdateSlice should end up as a stream operation.
    setupDirectHLOToFlowLegality(context, conversionTarget);
    populateHLOToFlowPatterns(context, conversionPatterns);
    setupStandardToFlowTensorLoadLegality(context, conversionTarget);

    conversionTarget.addLegalOp<linalg::GenericOp, linalg::IndexedGenericOp>();
    conversionTarget
        .markOpRecursivelyLegal<linalg::GenericOp, linalg::IndexedGenericOp>();
    populateStandardToFlowTensorLoadPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      getOperation().emitError()
          << "module is not in a compatible input format";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createPrePartitioningConversionPass() {
  return std::make_unique<PrePartitioningConversionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
