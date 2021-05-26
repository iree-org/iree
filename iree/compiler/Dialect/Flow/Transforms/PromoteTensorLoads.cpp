// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/StandardToFlow/ConvertStandardToFlow.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class PromoteTensorLoadsPass
    : public PromoteTensorLoadsBase<PromoteTensorLoadsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlowDialect, StandardOpsDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns(&getContext());

    conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();
    conversionTarget.addLegalDialect<StandardOpsDialect>();
    setupStandardToFlowTensorLoadLegality(context, conversionTarget);
    populateStandardToFlowTensorLoadPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createPromoteTensorLoadsPass() {
  return std::make_unique<PromoteTensorLoadsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
