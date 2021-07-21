// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

static bool isTFOp(Operation *op) {
  if (!op || !op->getDialect()) return false;
  StringRef opNamespace = op->getDialect()->getNamespace();
  return opNamespace == mlir::TF::TensorFlowDialect::getDialectNamespace() ||
         opNamespace == mlir::tf_executor::TensorFlowExecutorDialect::
                            getDialectNamespace() ||
         opNamespace ==
             mlir::tf_device::TensorFlowDeviceDialect::getDialectNamespace() ||
         opNamespace == mlir::tf_saved_model::TensorFlowSavedModelDialect::
                            getDialectNamespace();
}

class VerifyFullyConvertedPass
    : public PassWrapper<VerifyFullyConvertedPass, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect,
                    mlir::tf_executor::TensorFlowExecutorDialect,
                    mlir::tf_device::TensorFlowDeviceDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tf-verify-fully-converted";
  }

  StringRef getDescription() const override {
    return "Verifies that all TensorFlow frontend ops were converted and none "
           "remain";
  }

  // Validates that no TensorFlow frontends ops are in the function.
  void runOnFunction() override {
    DenseSet<Operation *> illegalOps;
    getFunction().walk([&](Operation *op) {
      if (isTFOp(op)) illegalOps.insert(op);
    });
    if (!illegalOps.empty()) {
      emitLegalizationErrors(getFunction().getLoc(), illegalOps);
      return signalPassFailure();
    }
  }

  // Emits debug information which includes the number of ops of each type which
  // failed to legalize.
  void emitLegalizationErrors(Location loc,
                              const DenseSet<Operation *> &nonlegalizedOps) {
    // Print op errors for each of the TensorFlow ops that still remain.
    std::map<StringRef, int> opNameCounts;
    for (Operation *nonlegalizedOp : nonlegalizedOps) {
      StringRef opName = nonlegalizedOp->getName().getStringRef();
      opNameCounts[opName]++;
      nonlegalizedOp->emitOpError()
          << ": unlegalized TensorFlow op still exists";
    }

    std::vector<std::string> errorMessages;
    errorMessages.reserve(opNameCounts.size());
    for (const auto &opInfo : opNameCounts) {
      errorMessages.push_back(
          llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
    }
    emitError(loc) << "The following Tensorflow operations still remain: \n"
                   << llvm::join(errorMessages, "\n") << "\n";
  }
};

static PassRegistration<VerifyFullyConvertedPass> pass;

std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
