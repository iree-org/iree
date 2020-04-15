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

class SerializeExecutablesPass
    : public PassWrapper<SerializeExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  SerializeExecutablesPass()
      : executableOptions_(getTargetOptionsFromFlags()) {}
  explicit SerializeExecutablesPass(TargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto targetOps = llvm::to_vector<4>(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>());
    for (auto targetOp : targetOps) {
      for (auto &targetBackend :
           matchTargetBackends({targetOp.target_backend().str()})) {
        // Ask the target backend to serialize the executable. Note that it may
        // create one or more hal.executable.binary ops in the case of
        // multi-architecture binaries.
        OpBuilder executableBuilder(targetOp);
        if (failed(targetBackend->serializeExecutable(targetOp,
                                                      executableBuilder))) {
          targetOp.emitError() << "failed to serialize op to target backend "
                               << targetOp.target_backend();
          return signalPassFailure();
        }
      }
      targetOp.erase();
    }
  }

 private:
  TargetOptions executableOptions_;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass(TargetOptions executableOptions) {
  return std::make_unique<SerializeExecutablesPass>(executableOptions);
}

static PassRegistration<SerializeExecutablesPass> pass(
    "iree-hal-serialize-executables",
    "Serializes hal.executable.target ops to hal.executable.binary ops");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
