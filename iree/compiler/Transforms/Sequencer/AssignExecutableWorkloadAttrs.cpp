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

#include "iree/compiler/IR/Sequencer/LLOps.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct WorkloadInfo {
  SmallVector<ElementsAttr, 4> staticWorkloads;
  SmallVector<Value *, 4> dynamicWorkloads;
};

// Finds all dispatches and records their workload attributes mapped by
// (executable ordinal, entry point ordinal).
llvm::StringMap<llvm::StringMap<WorkloadInfo>> gatherExecutableWorkloadInfos(
    ModuleOp moduleOp) {
  llvm::StringMap<llvm::StringMap<WorkloadInfo>> workloadInfos;
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    funcOp.walk([&](IREESeq::LL::DynamicDispatchOp op) {
      auto &workloadInfo =
          workloadInfos[op.getExecutable()][op.getEntryPoint()];
      workloadInfo.dynamicWorkloads.push_back(op.getWorkload());
    });
    funcOp.walk([&](IREESeq::LL::StaticDispatchOp op) {
      auto &workloadInfo =
          workloadInfos[op.getExecutable()][op.getEntryPoint()];
      for (auto existingWorkloadAttr : workloadInfo.staticWorkloads) {
        if (existingWorkloadAttr == op.getWorkload()) {
          return;  // Already present, ignore.
        }
      }
      workloadInfo.staticWorkloads.push_back(op.getWorkload());
    });
  }
  return workloadInfos;
}

// Adds attributes to the given executable entry point describing the workload
// info to the backends that will be processing them.
LogicalResult attributeExecutableEntryPointWorkload(
    FuncOp entryPointOp, const WorkloadInfo &workloadInfo) {
  if (!workloadInfo.dynamicWorkloads.empty()) {
    return entryPointOp.emitError() << "Dynamic workloads not yet supported";
  }
  if (workloadInfo.staticWorkloads.size() != 1) {
    return entryPointOp.emitError() << "Static workload sizes differ in shape";
  }

  // Easy because we just support static workloads now.
  // When this code is adapted to support dynamic workloads we'll want to put
  // a pair of attrs describing which dimensions may be static and which args
  // have the dynamic values to reference.
  entryPointOp.setAttr("iree.executable.workload",
                       workloadInfo.staticWorkloads.front());

  return success();
}

}  // namespace

class AssignExecutableWorkloadAttrsPass
    : public ModulePass<AssignExecutableWorkloadAttrsPass> {
 public:
  void runOnModule() override {
    Builder builder(getModule());

    // Find all dispatches and capture their workload information.
    // We store this information by executable and then entry point ordinal.
    auto executableWorkloadInfos = gatherExecutableWorkloadInfos(getModule());

    // Process each executable with the workload information.
    for (auto &executableIt : executableWorkloadInfos) {
      auto multiArchExecutableOp = cast<IREE::MultiArchExecutableOp>(
          getModule().lookupSymbol(executableIt.first()));
      for (auto executableOp :
           multiArchExecutableOp.getBlock().getOps<IREE::ExecutableOp>()) {
        for (auto &entryPointIt : executableIt.second) {
          auto funcOp = cast<FuncOp>(
              executableOp.getInnerModule().lookupSymbol(entryPointIt.first()));
          if (failed(attributeExecutableEntryPointWorkload(
                  funcOp, entryPointIt.second))) {
            return signalPassFailure();
          }
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>>
createAssignExecutableWorkloadAttrsPass() {
  return std::make_unique<AssignExecutableWorkloadAttrsPass>();
}

static PassRegistration<AssignExecutableWorkloadAttrsPass> pass(
    "iree-assign-executable-workload-attrs",
    "Assigns executable entrypoint workload attributes");

}  // namespace iree_compiler
}  // namespace mlir
