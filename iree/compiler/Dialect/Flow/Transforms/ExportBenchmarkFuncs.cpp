// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Clones each exported functions (including those just created) with
// placeholder constant inputs instead of arguments and removes the exported
// attribute from the old functions.
// The input are provided using util.globals.
class ExportBenchmarkFuncsPass
    : public ExportBenchmarkFuncsBase<ExportBenchmarkFuncsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather the functions we want to wrap for benchmarking and wrap them.
    // Since we are inserting new functions as part of this pass we must perform
    // the wrapping for only the inputs.
    SmallVector<mlir::func::FuncOp, 4> entryFuncOps;
    for (auto entryFuncOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (entryFuncOp.isPublic()) {
        entryFuncOps.push_back(entryFuncOp);
      }
    }
    for (auto entryFuncOp : entryFuncOps) {
      if (failed(createEntryPointBenchmarkFunc(moduleOp, entryFuncOp))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  IREE::Util::GlobalOp createDummyInputVariableOp(Location loc, Type inputType,
                                                  OpBuilder& moduleBuilder) {
    std::string baseName = "_benchmark_input_";
    std::string name = baseName + std::to_string(uniqueId++);
    auto initialValue = moduleBuilder.getZeroAttr(inputType);
    if (!initialValue) {
      mlir::emitError(loc) << "unsupported function argument type: "
                           << inputType;
      return {};
    }
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, name,
        /*isMutable=*/false, inputType, initialValue);
    globalOp.setPrivate();
    globalOp->setAttr("noinline", UnitAttr::get(moduleBuilder.getContext()));
    return globalOp;
  }

  LogicalResult createEntryPointBenchmarkFunc(mlir::ModuleOp moduleOp,
                                              mlir::func::FuncOp entryFuncOp) {
    OpBuilder moduleBuilder(&getContext());
    moduleBuilder.setInsertionPointAfter(entryFuncOp);

    // Create one dummy input variable per input.
    Location loc = entryFuncOp.getLoc();
    SmallVector<IREE::Util::GlobalOp, 4> dummyInputVariableOps;
    for (auto inputType : entryFuncOp.getFunctionType().getInputs()) {
      auto dummyVar = createDummyInputVariableOp(loc, inputType, moduleBuilder);
      if (!dummyVar) return failure();
      dummyInputVariableOps.push_back(dummyVar);
    }

    // Create a `() -> ()` entry point op the benchmark tool can run.
    std::string funcName = std::string(entryFuncOp.getName()) + "_benchmark";
    auto funcOp = moduleBuilder.create<mlir::func::FuncOp>(
        loc, funcName, moduleBuilder.getFunctionType({}, {}));
    funcOp.setPublic();
    funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    SmallVector<NamedAttribute> reflectionAttrs = {
        moduleBuilder.getNamedAttr("iree.benchmark",
                                   moduleBuilder.getStringAttr("entry")),
    };
    funcOp->setAttr("iree.reflection",
                    moduleBuilder.getDictionaryAttr(reflectionAttrs));
    Block* block = funcOp.addEntryBlock();

    // Call the existing function with dummy arguments.
    auto blockBuilder = OpBuilder::atBlockBegin(block);
    SmallVector<Value, 4> args;
    for (int i = 0, e = entryFuncOp.getNumArguments(); i < e; ++i) {
      args.push_back(blockBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
          loc, dummyInputVariableOps[i]));
    }
    auto callOp =
        blockBuilder.create<mlir::func::CallOp>(loc, entryFuncOp, args);

    // Sink all results with do_not_optimize to ensure that DCE does not
    // remove the call.
    for (auto result : callOp.getResults()) {
      blockBuilder.create<IREE::Util::DoNotOptimizeOp>(loc, result);
    }
    blockBuilder.create<mlir::func::ReturnOp>(loc);

    // Ensure the original function is not exported and not inlined.
    entryFuncOp->setAttr("noinline", moduleBuilder.getUnitAttr());
    entryFuncOp->removeAttr("iree.reflection");
    entryFuncOp.setPrivate();

    return success();
  }

  int uniqueId = 0;
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createExportBenchmarkFuncsPass() {
  return std::make_unique<ExportBenchmarkFuncsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
