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

class TranslateExecutablesPass
    : public PassWrapper<TranslateExecutablesPass,
                         OperationPass<IREE::HAL::ExecutableOp>> {
 public:
  TranslateExecutablesPass()
      : executableOptions_(getTargetOptionsFromFlags()) {}
  explicit TranslateExecutablesPass(TargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto targetOps = llvm::to_vector<4>(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>());
    for (auto targetOp : targetOps) {
      // TODO(#1036): this will be what we want the dynamic pass manager to
      // do for us: we want to nest all of the backend passes on a source op
      // that matches their target_backend pattern.
      for (auto &targetBackend :
           matchTargetBackends({targetOp.target_backend().str()})) {
        // Run the nested pass manager. This is effectively the same as
        // launching a new iree-opt, and as such won't integrate well with the
        // logging/pass instrumentation of the parent pass manager.
        PassManager targetPassManager(targetOp.getContext());
        applyPassManagerCLOptions(targetPassManager);
        targetBackend->buildTranslationPassPipeline(targetOp,
                                                    targetPassManager);
        if (failed(targetPassManager.run(targetOp.getInnerModule()))) {
          targetOp.emitError() << "failed to run translation of source "
                                  "executable to target executable for backend "
                               << targetOp.target_backend();
          return signalPassFailure();
        }
      }
    }
  }

 private:
  TargetOptions executableOptions_;
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createTranslateExecutablesPass(TargetOptions executableOptions) {
  return std::make_unique<TranslateExecutablesPass>(executableOptions);
}

static PassRegistration<TranslateExecutablesPass> pass(
    "iree-hal-translate-executables",
    "Translates hal.executable.target via the target backend pipelines");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
