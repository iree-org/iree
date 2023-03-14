// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "iree_tf_compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

class VerifyFullyConvertedPass
    : public PassWrapper<VerifyFullyConvertedPass,
                         OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-tf-verify-fully-converted";
  }

  StringRef getDescription() const override {
    return "Verifies that all TensorFlow frontend ops were converted and none "
           "remain";
  }

  // Validates that no TensorFlow frontends ops are in the function.
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target
        .addIllegalDialect<mlir::TF::TensorFlowDialect,
                           mlir::tf_executor::TensorFlowExecutorDialect,
                           mlir::tf_device::TensorFlowDeviceDialect,
                           mlir::tf_saved_model::TensorFlowSavedModelDialect>();
    if (failed(
            iree_compiler::verifyAllOperationsAreLegal(getOperation(), target)))
      return signalPassFailure();
  }
};

static PassRegistration<VerifyFullyConvertedPass> pass;

std::unique_ptr<OperationPass<func::FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
