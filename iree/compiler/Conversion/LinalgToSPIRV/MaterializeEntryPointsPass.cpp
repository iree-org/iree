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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-codegen-materialize-entry-points"

namespace mlir {
namespace iree_compiler {

namespace {

struct MaterializeEntryPointsPass
    : public PassWrapper<MaterializeEntryPointsPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  MaterializeEntryPointsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    auto existingEntryPointOps = llvm::to_vector<4>(
        targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>());

    ModuleOp moduleOp = targetOp.getInnerModule();
    spirv::ModuleOp spvModuleOp = *moduleOp.getOps<spirv::ModuleOp>().begin();
    auto spvEntryPointOps =
        llvm::to_vector<4>(spvModuleOp.getOps<spirv::EntryPointOp>());

    if (existingEntryPointOps.size() == spvEntryPointOps.size()) {
      return;
    }

    OpBuilder builder(moduleOp);
    auto templateEntryPointOp = existingEntryPointOps.front();
    for (size_t i = 1; i < spvEntryPointOps.size(); ++i) {
      auto spvEntryPointOp = spvEntryPointOps[i];
      auto clonedEntryPointOp = dyn_cast<IREE::HAL::ExecutableEntryPointOp>(
          builder.clone(*templateEntryPointOp.getOperation()));
      clonedEntryPointOp.sym_nameAttr(
          builder.getStringAttr(spvEntryPointOp.fn()));
    }
  }
};

};  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createMaterializeEntryPointsPass() {
  return std::make_unique<MaterializeEntryPointsPass>();
}

static PassRegistration<MaterializeEntryPointsPass> pass(
    "iree-codegen-materialize-entry-points",
    "Materialize HAL entry points for each spv.EntryPoint",
    [] { return std::make_unique<MaterializeEntryPointsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
