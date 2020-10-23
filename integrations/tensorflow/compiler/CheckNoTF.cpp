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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

class CheckNoTensorflow : public PassWrapper<CheckNoTensorflow, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, StandardOpsDialect>();
  }

  CheckNoTensorflow() = default;
  CheckNoTensorflow(const CheckNoTensorflow &) {}

  /// Validates that no TensorFlow frontends ops are in the function.
  void runOnFunction() override {
    auto op = getFunction();
    auto context = op.getContext();

    Dialect *dialect = context->getLoadedDialect("tf");
    DenseSet<Operation *> illegalOps;
    op.walk([&](Operation *op) {
      if (op->getDialect() == dialect) {
        illegalOps.insert(op);
      }
    });

    if (!illegalOps.empty()) {
      emitLegalizationErrors(op, illegalOps);
      return signalPassFailure();
    }
  }

  // Emits debug information which includes the number of ops of each type which
  // failed to legalize.
  void emitLegalizationErrors(Operation *op,
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
    Location loc = op->getLoc();
    emitError(loc) << "The following Tensorflow operations still remain: \n"
                   << llvm::join(errorMessages, "\n") << "\n";
  }
};

static PassRegistration<CheckNoTensorflow> pass(
    "iree-check-no-tf", "Check that no TensorFlow frontend ops remain");
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createCheckNoTF() {
  return std::make_unique<CheckNoTensorflow>();
}

}  // namespace iree_compiler
}  // namespace mlir
