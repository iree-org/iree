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

#include "iree_tf_compiler/TF/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

class StripAssertsPass
    : public PassWrapper<StripAssertsPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    DenseSet<Operation *> assertOps;
    funcOp.walk([&](Operation *op) {
      if (isa<mlir::TF::AssertOp>(op)) {
        assertOps.insert(op);
      }
    });

    for (Operation *assertOp : assertOps) {
      assertOp->erase();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createStripAssertsPass() {
  return std::make_unique<StripAssertsPass>();
}

static PassRegistration<StripAssertsPass> funcPass("iree-tf-strip-asserts",
                                                   "Remove tf.Assert ops");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
