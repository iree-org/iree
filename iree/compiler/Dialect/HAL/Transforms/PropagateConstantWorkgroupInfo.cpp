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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class PropagateConstantWorkgroupInfoPass
    : public PassWrapper<PropagateConstantWorkgroupInfoPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  void runOnOperation() override {
    auto targetOp = getOperation();

    SymbolTable targetSymbolTable(targetOp);
    for (auto funcOp : targetOp.getInnerModule().getOps<FuncOp>()) {
      auto entryPointOp =
          targetSymbolTable.lookup<IREE::HAL::ExecutableEntryPointOp>(
              funcOp.getName());
      if (!entryPointOp) continue;
      if (!entryPointOp.workgroup_size().hasValue()) continue;
      auto workgroupSizeAttr = entryPointOp.workgroup_sizeAttr();
      auto workgroupSizeOps = llvm::to_vector<4>(
          funcOp.getOps<IREE::HAL::InterfaceWorkgroupSizeOp>());
      for (auto workgroupSizeOp : workgroupSizeOps) {
        OpBuilder builder(workgroupSizeOp);
        auto dimValue = builder.createOrFold<ConstantIndexOp>(
            workgroupSizeOp.getLoc(),
            workgroupSizeAttr[workgroupSizeOp.dimension().getZExtValue()]
                .cast<IntegerAttr>()
                .getInt());
        workgroupSizeOp.replaceAllUsesWith(dimValue);
        workgroupSizeOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createPropagateConstantWorkgroupInfoPass() {
  return std::make_unique<PropagateConstantWorkgroupInfoPass>();
}

static PassRegistration<PropagateConstantWorkgroupInfoPass> pass(
    "iree-hal-propagate-constant-workgroup-info",
    "Propagates constant hal.interface.workgroup.* queries when known");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
