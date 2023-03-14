// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "iree_tf_compiler/Utils/ConversionUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

class VerifyFullyConvertedPass
    : public VerifyFullyConvertedBase<VerifyFullyConvertedPass> {
 public:

  // Validates that no TFLite frontends ops are in the function.
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalDialect<mlir::TFL::TensorFlowLiteDialect>();
    target.addIllegalOp<mlir::UnrealizedConversionCastOp>();
    if (failed(
            iree_compiler::verifyAllOperationsAreLegal(getOperation(), target)))
      return signalPassFailure();
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
