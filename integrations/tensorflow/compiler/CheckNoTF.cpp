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

class CheckNoTF : public PassWrapper<CheckNoTF, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, StandardOpsDialect>();
  }

  CheckNoTF() = default;
  CheckNoTF(const CheckNoTF &) {}

  /// Performs the lowering to XLA dialect.
  void runOnFunction() override {
    auto op = getFunction();  
    auto context = op.getContext();

    Dialect* dialect = context->getLoadedDialect("tf");
    DenseSet<Operation *> illegalOps;
    op.walk([&](Operation *op) {
        if (op->getDialect() == dialect) {
            illegalOps.insert(op);
        }
    });

    if (!illegalOps.empty()) {
        EmitLegalizationErrors(op, illegalOps);
        return signalPassFailure();
    }
  }

  // Emits debug information which includes the number of ops of each type which
  // failed to legalize.
  void EmitLegalizationErrors(Operation *op,
                              const DenseSet<Operation *> &nonlegalized_ops) {
    // Track the legalization failures by mapping op name to information about
    // that failure: the number of unlegalized occurrences of the op, and one
    // example operation that failed.
    std::map<StringRef, std::pair<int, Operation *>> op_name_to_error_info;
    DenseSet<Operation *> error_ops;
    for (Operation *nonlegalized_op : nonlegalized_ops) {
      // Increment count of this legalization failure.
      StringRef op_name = nonlegalized_op->getName().getStringRef();
      // If this emplace is successful, it's the first time we've encountered
      // this op type. Initialize count to 0 so that after increment, it is 1.
      auto insertion_result = op_name_to_error_info.emplace(
          op_name, std::make_pair(0, nonlegalized_op));
      ++insertion_result.first->second.first;
    }
    std::vector<std::string> error_messages;
    error_messages.reserve(op_name_to_error_info.size());
    for (const auto &op_info : op_name_to_error_info) {
      error_messages.push_back(
          llvm::formatv("{0} (count: {1})", op_info.first, op_info.second.first));
    }
    Location loc = op->getLoc();
    emitError(loc) << "The following operations cannot be legalized: "
                   << llvm::join(error_messages, "; ")
                   << ". These legalization failure(s) may be due to missing TF "
                      "to HLO lowerings and/or unsupported attributes, etc.";
    // Emit more information about the missing ops. This error message
    // contains useful details beyond the op name (input and output shapes,
    // attributes, etc.).
    for (const auto &op_info : op_name_to_error_info) {
      op_info.second.second->emitOpError() << "is not legalizable";
    }
  }
};

static PassRegistration<CheckNoTF> pass(
    "iree-check-no-tf", "Check that no TF remains");
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createCheckNoTF() {
  return std::make_unique<CheckNoTF>();
}

}  // namespace iree_compiler
}  // namespace mlir
