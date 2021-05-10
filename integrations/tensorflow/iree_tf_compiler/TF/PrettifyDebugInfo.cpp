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

#include "iree_tf_compiler/TF/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

class PrettifyDebugInfoPass
    : public PassWrapper<PrettifyDebugInfoPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    // TODO: Finish algorithm for simplifying TF debug info.
    // auto moduleOp = getOperation();
    // moduleOp.walk([&](Operation *op) {
    //   Location loc = op->getLoc();
    //   if (auto callSite = loc.dyn_cast<CallSiteLoc>()) {
    //     callSite.getCallee().dump();
    //   }
    // });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPrettifyDebugInfoPass() {
  return std::make_unique<PrettifyDebugInfoPass>();
}

static PassRegistration<PrettifyDebugInfoPass> modulePass(
    "iree-tf-prettify-debug-info",
    "Simplifies TF debug info to make it easier to look at");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
