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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class CreateFuncsToInvokeExecOpsPass
    : public PassWrapper<CreateFuncsToInvokeExecOpsPass,
                         OperationPass<ModuleOp>> {
 public:
  CreateFuncsToInvokeExecOpsPass() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    Location loc = moduleOp.getLoc();
    auto execOps = moduleOp.getOps<IREE::Flow::ExecutableOp>();
    for (auto execOp : execOps) {
      for (auto& op : execOp.getBlock()) {
        if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
          auto execFuncOp = execOp.getInnerModule().lookupSymbol<FuncOp>(
              dispatchEntryOp.function_ref());
          std::string funcName = std::string(execFuncOp.getName()) + "_entry";
          auto funcType =
              builder.getFunctionType({}, execFuncOp.getType().getResults());
          auto funcOp =
              builder.create<FuncOp>(moduleOp.getLoc(), funcName, funcType);
          funcOp.setAttr("iree.module.export", UnitAttr::get(&getContext()));
          Block* block = funcOp.addEntryBlock();
          auto blockBuilder = OpBuilder(block, block->begin());
          SmallVector<Value, 4> args;
          for (auto inputType : execFuncOp.getType().getInputs()) {
            // TODO(hanchung): Use non-zero or random values as inputs.
            auto attr = blockBuilder.getZeroAttr(inputType);
            auto cst = blockBuilder.create<ConstantOp>(moduleOp.getLoc(),
                                                       inputType, attr);
            args.push_back(
                blockBuilder.create<IREE::DoNotOptimizeOp>(loc, ValueRange{cst})
                    .getResult(0));
          }
          auto dummyWorkload = blockBuilder.create<ConstantIndexOp>(loc, 0);
          auto dispatchOp = blockBuilder.create<DispatchOp>(
              loc, dispatchEntryOp, dummyWorkload, funcType.getResults(), args);
          blockBuilder.create<mlir::ReturnOp>(loc, dispatchOp.getResults());
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createCreateFuncsToInvokeExecOpsPass() {
  return std::make_unique<CreateFuncsToInvokeExecOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
