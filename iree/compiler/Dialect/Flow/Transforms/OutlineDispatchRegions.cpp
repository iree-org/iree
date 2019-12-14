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
#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Converts a dispatch_region into a dispatch to the outlined region function.
LogicalResult convertToDispatchOp(DispatchRegionOp regionOp,
                                  ExecutableOp executableOp,
                                  DispatchEntryOp entryPointOp,
                                  FuncOp outlinedFuncOp) {
  // Insert at the same place as the original region.
  OpBuilder builder(regionOp);

  // Create the dispatch op to the executable function.
  auto dispatchOp = builder.create<DispatchOp>(
      regionOp.getLoc(), executableOp.getName(), entryPointOp.getName(),
      regionOp.workload(), outlinedFuncOp.getType().getResults(),
      regionOp.args());

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i)->replaceAllUsesWith(dispatchOp.getResult(i));
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Outlines a dispatch region into a flow.executable.
LogicalResult outlineDispatchRegion(
    DispatchRegionOp regionOp, int outlinedRegionOrdinal,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 8> operandTypes(
      Operation::operand_type_range{regionOp.args()});
  SmallVector<Type, 8> resultTypes(regionOp.getResultTypes());
  auto functionType =
      FunctionType::get(operandTypes, resultTypes, regionOp.getContext());

  // Create the executable with the region cloned into it.
  ExecutableOp executableOp;
  FuncOp outlinedFuncOp;
  std::tie(executableOp, outlinedFuncOp) = createRegionExecutable(
      regionOp, functionType,
      "_dispatch_" + std::to_string(outlinedRegionOrdinal),
      dispatchableFuncOps);

  // Add dispatch export pointing at the function.
  OpBuilder builder(executableOp.body());
  auto entryPointOp = builder.create<DispatchEntryOp>(
      regionOp.getLoc(), builder.getStringAttr(outlinedFuncOp.getName()),
      builder.getSymbolRefAttr(outlinedFuncOp), DenseIntElementsAttr{},
      DenseIntElementsAttr{});

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertToDispatchOp(regionOp, executableOp, entryPointOp,
                             outlinedFuncOp);
}

}  // namespace

class OutlineDispatchRegionsPass
    : public ModulePass<OutlineDispatchRegionsPass> {
 public:
  OutlineDispatchRegionsPass() = default;
  OutlineDispatchRegionsPass(
      std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps)
      : dispatchableFuncOps_(std::move(dispatchableFuncOps)) {}

  void runOnModule() override {
    // TODO(benvanik): replace with a pattern rewriter?
    auto funcOps = llvm::to_vector<32>(getModule().getOps<FuncOp>());
    for (auto funcOp : funcOps) {
      // Outline all of the dispatch regions ops in this function.
      SmallVector<DispatchRegionOp, 8> dispatchRegionOps;
      funcOp.walk(
          [&](DispatchRegionOp op) { dispatchRegionOps.push_back(op); });
      for (int i = 0; i < dispatchRegionOps.size(); ++i) {
        if (failed(outlineDispatchRegion(dispatchRegionOps[i], i,
                                         *dispatchableFuncOps_))) {
          return signalPassFailure();
        }
      }
    }
  }

 private:
  std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps_;
};

std::unique_ptr<OpPassBase<ModuleOp>> createOutlineDispatchRegionsPass(
    std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps) {
  return std::make_unique<OutlineDispatchRegionsPass>(
      std::move(dispatchableFuncOps));
}

static PassRegistration<OutlineDispatchRegionsPass> pass(
    "iree-flow-outline-dispatch-regions",
    "Outlines dispatch regions into standalone functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
