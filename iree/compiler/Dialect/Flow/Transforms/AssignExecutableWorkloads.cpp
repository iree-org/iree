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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
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
namespace IREE {
namespace Flow {

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
    funcOp.walk([&](DispatchOp op) {
      auto &workloadInfo = workloadInfos[op.executable()][op.entry_point()];
      if (auto constantOp =
              dyn_cast<ConstantOp>(op.workload()->getDefiningOp())) {
        for (auto existingWorkloadAttr : workloadInfo.staticWorkloads) {
          if (existingWorkloadAttr == constantOp.value()) {
            return;  // Already present, ignore.
          }
        }
        workloadInfo.staticWorkloads.push_back(
            constantOp.value().cast<ElementsAttr>());
      } else {
        workloadInfo.dynamicWorkloads.push_back(op.workload());
      }
    });
  }
  return workloadInfos;
}

// Adds attributes to the given executable entry point describing the workload
// info to the backends that will be processing them.
LogicalResult attributeExecutableEntryPointWorkload(
    Operation *entryPointOp, const WorkloadInfo &workloadInfo) {
  if (!workloadInfo.dynamicWorkloads.empty()) {
    return entryPointOp->emitError() << "dynamic workloads not yet supported";
  }
  if (workloadInfo.staticWorkloads.size() != 1) {
    return entryPointOp->emitError() << "static workload sizes differ in shape";
  }

  // Easy because we just support static workloads now.
  // When this code is adapted to support dynamic workloads we'll want to put
  // a pair of attrs describing which dimensions may be static and which args
  // have the dynamic values to reference.
  entryPointOp->setAttr("workload", workloadInfo.staticWorkloads.front());

  // Hardwire workgroup size to {32, 1, 1}
  SmallVector<int32_t, 3> workGroupInfo = {32, 1, 1};
  auto workGroupAttr = DenseIntElementsAttr::get<int32_t>(
      RankedTensorType::get(3,
                            IntegerType::get(32, entryPointOp->getContext())),
      workGroupInfo);
  entryPointOp->setAttr("workgroup_size", workGroupAttr);
  return success();
}

}  // namespace

class AssignExecutableWorkloadsPass
    : public ModulePass<AssignExecutableWorkloadsPass> {
 public:
  void runOnModule() override {
    Builder builder(getModule());

    // Find all dispatches and capture their workload information.
    // We store this information by executable and then entry point ordinal.
    auto executableWorkloadInfos = gatherExecutableWorkloadInfos(getModule());

    // Process each executable with the workload information.
    SymbolTable symbolTable(getModule());
    for (auto &executableIt : executableWorkloadInfos) {
      auto executableOp =
          symbolTable.lookup<ExecutableOp>(executableIt.first());
      for (auto &entryPointIt : executableIt.second) {
        auto entryPointOp = executableOp.lookupSymbol(entryPointIt.first());
        if (failed(attributeExecutableEntryPointWorkload(
                entryPointOp, entryPointIt.second))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createAssignExecutableWorkloadsPass() {
  return std::make_unique<AssignExecutableWorkloadsPass>();
}

static PassRegistration<AssignExecutableWorkloadsPass> pass(
    "iree-flow-assign-executable-workloads",
    "Assigns executable entrypoint workload attributes");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
