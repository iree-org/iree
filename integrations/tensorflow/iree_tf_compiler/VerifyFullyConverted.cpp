// Copyright 2020 Google LLC
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

static PassRegistration<VerifyFullyConvertedPass> pass(
    "iree-tf-verify-fully-converted",
    "Verifies that all TensorFlow frontend ops were converted and none remain");

std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
