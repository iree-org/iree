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
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(GH-67): move workgroup size determination to backends.
// Returns an (x,y,z) workgroup size for the given |targetFuncOp|.
// This is pure heuristics until we support dynamic/varying workgroup sizes.
static std::array<int32_t, 3> guessWorkGroupSize(
    IREE::Flow::DispatchEntryOp entryOp, FuncOp targetFuncOp) {
  for (auto& block : targetFuncOp.getBlocks()) {
    if (!block.getOps<xla_hlo::DotOp>().empty()) {
      // A special dot kernel. This has a fixed workgroup size based on the
      // hand-written shader.
      return {16, 16, 1};
    } else if (!block.getOps<xla_hlo::ConvOp>().empty()) {
      // Matches hard-coded assumptions in the conv2d_nhwc hand-written
      // shader.
      return {1, 1, 1};
    }
  }
  return {32, 1, 1};
}

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
      targetBackends.append(getExecutableTargetRegistry().keys().begin(),
                            getExecutableTargetRegistry().keys().end());
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
        llvm::to_vector<32>(getModule().getOps<IREE::Flow::ExecutableOp>());
    for (auto sourceOp : executableOps) {
      // Create the op that will contain the translated executables.
      OpBuilder builder(getModule().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto targetOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());

      // Annotate the entry points.
      addEntryPointOps(sourceOp, targetOp);

      // Translate for each backend. Variants will be added to the targetOp.
      for (auto targetBackend : targetBackends) {
        auto targetFn = getExecutableTargetRegistry().find(targetBackend);
        if (targetFn == getExecutableTargetRegistry().end()) {
          sourceOp.emitError()
              << "target backend '" << targetBackend << "' unavailable";
          return signalPassFailure();
        }

        // Perform translation using the registered translation function.
        if (failed(targetFn->second(sourceOp, targetOp, executableOptions_))) {
          sourceOp.emitError()
              << "failed translation to target " << targetBackend;
          return signalPassFailure();
        }
      }

      // Erase the original flow.executable.
      sourceOp.erase();
    }
  }

 private:
  // Adds the entry point ops with assigned ordinals for each entry function.
  void addEntryPointOps(IREE::Flow::ExecutableOp sourceOp,
                        IREE::HAL::ExecutableOp targetOp) {
    OpBuilder builder(targetOp.getContext());
    builder.setInsertionPointToStart(&targetOp.getBlock());
    int nextOrdinal = 0;
    for (auto& op : sourceOp.getBlock()) {
      if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
        // Hardwire workgroup size based on the contents.
        // TODO(GH-67): move workgroup size determination to backends.
        auto targetFuncOp = sourceOp.getInnerModule().lookupSymbol<FuncOp>(
            dispatchEntryOp.function_ref());
        auto workGroupSize = guessWorkGroupSize(dispatchEntryOp, targetFuncOp);
        auto workGroupSizeAttr = DenseIntElementsAttr::get(
            VectorType::get(3, builder.getIntegerType(32)), workGroupSize);

        builder.create<IREE::HAL::ExecutableEntryPointOp>(
            dispatchEntryOp.getLoc(),
            builder.getStringAttr(dispatchEntryOp.sym_name()),
            builder.getI32IntegerAttr(nextOrdinal++), workGroupSizeAttr);
      } else if (auto reductionEntryOp =
                     dyn_cast<IREE::Flow::ReductionEntryOp>(op)) {
        builder.create<IREE::HAL::ExecutableEntryPointOp>(
            reductionEntryOp.getLoc(),
            builder.getStringAttr(reductionEntryOp.sym_name()),
            builder.getI32IntegerAttr(nextOrdinal++),
            DenseIntElementsAttr::get(
                VectorType::get({3}, builder.getIntegerType(32)),
                ArrayRef<int32_t>{1, 1, 1}));
      }
    }
  }

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
