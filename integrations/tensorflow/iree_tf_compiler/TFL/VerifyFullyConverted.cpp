// Copyright 2021 Google LLC
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

#include "iree_tf_compiler/TFL/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

static bool isTFLOp(Operation *op) {
  if (!op || !op->getDialect()) return false;
  StringRef opNamespace = op->getDialect()->getNamespace();
  return opNamespace == mlir::TFL::TensorFlowLiteDialect::getDialectNamespace();
}

class VerifyFullyConvertedPass
    : public PassWrapper<VerifyFullyConvertedPass, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  // Validates that no TFLite frontends ops are in the function.
  void runOnFunction() override {
    DenseSet<Operation *> illegalOps;
    getFunction().walk([&](Operation *op) {
      if (isTFLOp(op)) illegalOps.insert(op);
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
    // Print op errors for each of the TFLite ops that still remain.
    std::map<StringRef, int> opNameCounts;
    for (Operation *nonlegalizedOp : nonlegalizedOps) {
      StringRef opName = nonlegalizedOp->getName().getStringRef();
      opNameCounts[opName]++;
      nonlegalizedOp->emitOpError() << ": unlegalized TFLite op still exists";
    }

    std::vector<std::string> errorMessages;
    errorMessages.reserve(opNameCounts.size());
    for (const auto &opInfo : opNameCounts) {
      errorMessages.push_back(
          llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
    }
    emitError(loc) << "The following TFLite operations still remain: \n"
                   << llvm::join(errorMessages, "\n") << "\n";
  }
};

static PassRegistration<VerifyFullyConvertedPass> pass(
    "iree-tflite-verify-fully-converted",
    "Verifies that all TFLite frontend ops were converted and none remain");

std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
