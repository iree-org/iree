// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(ataei): Better way of listing default backends translation.
static llvm::StringSet<> excludedBackends = {"llvm-ir"};

class TranslateExecutablesPass : public ModulePass<TranslateExecutablesPass> {
 public:
  TranslateExecutablesPass()
      : executableOptions_(getExecutableTargetOptionsFromFlags()) {}
  explicit TranslateExecutablesPass(ExecutableTargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void runOnModule() override {
    // Get the target backends we want to translate executables into.
    SmallVector<std::string, 4> targetBackends;
    if (executableOptions_.targets.empty()) {
      for (const auto backend : getExecutableTargetRegistry().keys()) {
        if (!excludedBackends.count(backend)) {
          targetBackends.emplace_back(backend);
        }
      }
    } else {
      for (auto targetName : executableOptions_.targets) {
        auto backendNames = matchExecutableTargetNames(targetName);
        targetBackends.append(backendNames.begin(), backendNames.end());
      }
    }
    if (targetBackends.empty()) {
      auto diagnostic = getModule().emitError();
      diagnostic
          << "no target backends available for executable translation; ensure "
          << "they are linked in and the target options are properly "
          << "specified. requested = [ ";
      for (const auto& target : executableOptions_.targets) {
        diagnostic << "'" << target << "' ";
      }
      diagnostic << "], available = [ ";
      for (const auto& target : getExecutableTargetRegistry().keys()) {
        diagnostic << "'" << target << "' ";
      }
      diagnostic << "]";
      return signalPassFailure();
    }

    // Translate all executables to all backends.
    // When we want heterogenous support we'll do this differently - possibly
    // even earlier on in the flow dialect (at least, deciding).
    auto executableOps =
        llvm::to_vector<32>(getModule().getOps<IREE::HAL::ExecutableOp>());
    for (auto executableOp : executableOps) {
      // Translate for each backend. Variants will be added to the executableOp.
      for (auto targetBackend : targetBackends) {
        auto targetFn = getExecutableTargetRegistry().find(targetBackend);
        if (targetFn == getExecutableTargetRegistry().end()) {
          executableOp.emitError()
              << "target backend '" << targetBackend << "' unavailable";
          return signalPassFailure();
        }

        // Perform translation using the registered translation function.
        if (failed(targetFn->second(executableOp, executableOptions_))) {
          executableOp.emitError()
              << "failed translation to target " << targetBackend;
          return signalPassFailure();
        }
      }

      // Erase the original flow.executable.
      auto sourceOp = executableOp.getSourceOp();
      sourceOp.erase();
    }
  }

 private:
  ExecutableTargetOptions executableOptions_;
};

std::unique_ptr<OpPassBase<ModuleOp>> createTranslateExecutablesPass(
    ExecutableTargetOptions executableOptions) {
  return std::make_unique<TranslateExecutablesPass>(
      executableOptions);  // NOLINT
}

static PassRegistration<TranslateExecutablesPass> pass(
    "iree-hal-translate-executables",
    "Translates flow.executable ops to hal.executable ops");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
