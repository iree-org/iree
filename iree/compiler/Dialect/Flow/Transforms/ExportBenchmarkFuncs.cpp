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
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Exports two kind of benchmark functions:
//   - Creates exported functions to invoke each executable op.
//   - Clones each exported functions (including those just created) with
//     placeholder constant inputs instead of arguments and removes the
//     exported attribute from the old functions.
// The input are provided using flow.variable and flow.lookup.
class ExportBenchmarkFuncsPass
    : public ExportBenchmarkFuncsBase<ExportBenchmarkFuncsPass> {
 public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Gather the functions we want to wrap for benchmarking and wrap them.
    // Since we are inserting new functions as part of this pass we must perform
    // the wrapping for only the inputs.
    SmallVector<FuncOp, 4> entryFuncOps;
    for (auto entryFuncOp : moduleOp.getOps<FuncOp>()) {
      if (entryFuncOp->getAttr("iree.module.export")) {
        entryFuncOps.push_back(entryFuncOp);
      }
    }
    for (auto entryFuncOp : entryFuncOps) {
      createEntryPointBenchmarkFunc(moduleOp, entryFuncOp);
    }

    // Create one benchmark function per entry point in each flow.executable.
    for (auto executableOp : moduleOp.getOps<IREE::Flow::ExecutableOp>()) {
      createExecutableBenchmarkFunc(moduleOp, executableOp);
    }
  }

 private:
  IREE::Flow::VariableOp createDummyInputVariableOp(Location loc,
                                                    Type inputType,
                                                    OpBuilder& moduleBuilder) {
    std::string baseName = "_benchmark_input_";
    std::string name = baseName + std::to_string(uniqueId++);
    auto variableOp = moduleBuilder.create<VariableOp>(
        loc, name,
        /*isMutable=*/false, inputType, moduleBuilder.getZeroAttr(inputType));
    variableOp.setPrivate();
    variableOp->setAttr("noinline", UnitAttr::get(moduleBuilder.getContext()));
    return variableOp;
  }

  void createEntryPointBenchmarkFunc(ModuleOp moduleOp, FuncOp entryFuncOp) {
    OpBuilder moduleBuilder(&getContext());
    moduleBuilder.setInsertionPointAfter(entryFuncOp);

    // Create one dummy input variable per input.
    Location loc = entryFuncOp.getLoc();
    SmallVector<IREE::Flow::VariableOp, 4> dummyInputVariableOps;
    for (auto inputType : entryFuncOp.getType().getInputs()) {
      dummyInputVariableOps.push_back(
          createDummyInputVariableOp(loc, inputType, moduleBuilder));
    }

    // Create a `() -> ()` entry point op the benchmark tool can run.
    std::string funcName = std::string(entryFuncOp.getName()) + "_benchmark";
    auto funcOp = moduleBuilder.create<FuncOp>(
        loc, funcName, moduleBuilder.getFunctionType({}, {}));
    funcOp.setPublic();
    funcOp->setAttr("iree.module.export", moduleBuilder.getUnitAttr());
    funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    SmallVector<NamedAttribute> reflectionAttrs = {
        moduleBuilder.getNamedAttr("benchmark",
                                   moduleBuilder.getStringAttr("entry")),
    };
    funcOp->setAttr("iree.reflection",
                    moduleBuilder.getDictionaryAttr(reflectionAttrs));
    Block* block = funcOp.addEntryBlock();

    // Call the existing function with dummy arguments.
    auto blockBuilder = OpBuilder::atBlockBegin(block);
    SmallVector<Value, 4> args;
    for (int i = 0, e = entryFuncOp.getNumArguments(); i < e; ++i) {
      args.push_back(blockBuilder.createOrFold<IREE::Flow::VariableLoadOp>(
          loc, dummyInputVariableOps[i]));
    }
    auto callOp = blockBuilder.create<mlir::CallOp>(loc, entryFuncOp, args);

    // Sink all results with do_not_optimize to ensure that DCE does not
    // remove the call.
    for (auto result : callOp.getResults()) {
      blockBuilder.create<IREE::DoNotOptimizeOp>(loc, result);
    }
    blockBuilder.create<mlir::ReturnOp>(loc);

    // Ensure the original function is not exported and not inlined.
    entryFuncOp->setAttr("noinline", moduleBuilder.getUnitAttr());
    entryFuncOp->removeAttr("iree.module.export");
    entryFuncOp->removeAttr("iree.reflection");
    entryFuncOp.setPrivate();
  }

  void createExecutableBenchmarkFunc(ModuleOp moduleOp,
                                     IREE::Flow::ExecutableOp executableOp) {
    OpBuilder moduleBuilder(&getContext());
    moduleBuilder.setInsertionPointAfter(executableOp);
    for (auto& op : executableOp.getBlock()) {
      auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op);
      if (!dispatchEntryOp) continue;
      auto execFuncOp = executableOp.getInnerModule().lookupSymbol<FuncOp>(
          dispatchEntryOp.function_ref());
      Location loc = execFuncOp.getLoc();

      // Create one dummy input variable per input.
      SmallVector<IREE::Flow::VariableOp, 4> dummyInputVariableOps;
      for (auto inputType : execFuncOp.getType().getInputs()) {
        dummyInputVariableOps.push_back(
            createDummyInputVariableOp(loc, inputType, moduleBuilder));
      }

      // Create a `() -> ()` entry point op the benchmark tool can run.
      std::string funcName = std::string(execFuncOp.getName()) + "_benchmark";
      auto funcType = moduleBuilder.getFunctionType({}, {});
      auto funcOp = moduleBuilder.create<FuncOp>(loc, funcName, funcType);
      funcOp->setAttr("iree.module.export", moduleBuilder.getUnitAttr());
      funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
      SmallVector<NamedAttribute> reflectionAttrs = {
          moduleBuilder.getNamedAttr("benchmark",
                                     moduleBuilder.getStringAttr("dispatch")),
      };
      funcOp->setAttr("iree.reflection",
                      moduleBuilder.getDictionaryAttr(reflectionAttrs));
      Block* block = funcOp.addEntryBlock();

      // Build the body of the FuncOp.
      auto blockBuilder = OpBuilder(block, block->begin());
      SmallVector<Value, 4> args;
      for (auto variableOp : dummyInputVariableOps) {
        args.push_back(blockBuilder.createOrFold<IREE::Flow::VariableLoadOp>(
            loc, variableOp));
      }

      // TODO(hanchung): Use a real workload instead? We can probably
      // calculate the workload from the results.
      auto dummyWorkload = blockBuilder.create<ConstantIndexOp>(loc, 0);
      auto dispatchOp = blockBuilder.create<DispatchOp>(
          loc, dispatchEntryOp, ValueRange{dummyWorkload},
          execFuncOp.getType().getResults(), ValueRange{}, args, ValueRange{},
          ArrayRef<int64_t>{});

      // Sink all results with do_not_optimize to ensure that DCE does not
      // remove the dispatch.
      for (auto result : dispatchOp.getResults()) {
        blockBuilder.create<IREE::DoNotOptimizeOp>(loc, result);
      }
      blockBuilder.create<mlir::ReturnOp>(loc);
    }
  }

  int uniqueId = 0;
};

std::unique_ptr<OperationPass<ModuleOp>> createExportBenchmarkFuncsPass() {
  return std::make_unique<ExportBenchmarkFuncsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
