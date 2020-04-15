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

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LinkExecutablesPass
    : public PassWrapper<LinkExecutablesPass, OperationPass<mlir::ModuleOp>> {
 public:
  LinkExecutablesPass() : executableOptions_(getTargetOptionsFromFlags()) {}
  explicit LinkExecutablesPass(TargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    for (auto &targetBackend :
         matchTargetBackends(executableOptions_.targets)) {
      // Ask the target backend to link all executables it wants.
      if (failed(targetBackend->linkExecutables(moduleOp))) {
        moduleOp.emitError() << "failed to link executables for target backend "
                             << targetBackend->name();
        return signalPassFailure();
      }
    }
  }

 private:
  TargetOptions executableOptions_;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkExecutablesPass(
    TargetOptions executableOptions) {
  return std::make_unique<LinkExecutablesPass>(executableOptions);
}

static PassRegistration<LinkExecutablesPass> pass(
    "iree-hal-link-executables",
    "Links together hal.executables depending on target backend rules");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
