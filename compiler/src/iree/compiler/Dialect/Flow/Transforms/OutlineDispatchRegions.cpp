// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {
namespace {

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

// Creates a flow.executable out of a set of functions, pulling in all other
// functions reachable by the provided functions.
static ExecutableOp createExecutable(Location loc, StringRef executableName,
                                     ArrayRef<mlir::func::FuncOp> funcOps,
                                     mlir::ModuleOp parentModuleOp) {
  assert(!funcOps.empty() && "must have at least one entry function");

  // Create the executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  auto executableOp =
      parentModuleBuilder.create<IREE::Flow::ExecutableOp>(loc, executableName);

  // Create the inner ModuleOp that contains the original functions. We need
  // to provide this shim as some ops (like std.call) look for the
  // containing module to provide symbol resolution.
  OpBuilder executableBuilder(executableOp);
  executableBuilder.setInsertionPointToStart(&executableOp.getBlock());
  auto innerModule = executableBuilder.create<mlir::ModuleOp>(loc);
  for (auto funcOp : funcOps) {
    innerModule.push_back(funcOp);
  }

  // Copy all reachable functions into the executable.
  // Linker passes may dedupe these later on.
  OpBuilder innerModuleBuilder = OpBuilder::atBlockEnd(innerModule.getBody());
  innerModuleBuilder.setInsertionPoint(innerModule.getBody(),
                                       ++innerModule.getBody()->begin());

  return executableOp;
}

// Converts a dispatch region op into a dispatch op to the outlined region.
static LogicalResult convertDispatchWorkgroupsToDispatchOp(
    IREE::Flow::DispatchWorkgroupsOp dispatchWorkgroupsOp,
    IREE::Flow::ExecutableOp executableOp,
    IREE::Flow::ExecutableExportOp exportOp) {
  // Insert at the same place as the original region.
  OpBuilder builder(dispatchWorkgroupsOp);

  // Create the dispatch op to the executable function.
  // Note that we copy the tied operand indices from the workgroups op - it
  // lines up 1:1 with the dispatch once we've outlined things.
  auto dispatchOp = builder.create<IREE::Flow::DispatchOp>(
      dispatchWorkgroupsOp.getLoc(), exportOp,
      dispatchWorkgroupsOp.getWorkload(), dispatchWorkgroupsOp.getResultTypes(),
      dispatchWorkgroupsOp.getResultDims(), dispatchWorkgroupsOp.getArguments(),
      dispatchWorkgroupsOp.getArgumentDims(),
      dispatchWorkgroupsOp.getTiedOperandsAttr());
  dispatchOp->setDialectAttrs(dispatchWorkgroupsOp->getDialectAttrs());

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < dispatchWorkgroupsOp.getNumResults(); ++i) {
    dispatchWorkgroupsOp.getResult(i).replaceAllUsesWith(
        dispatchOp.getResult(i));
  }

  return success();
}

// Converts a dispatch region body to a free-floating function.
static mlir::func::FuncOp
createWorkgroupFunc(Location loc, StringRef functionName, Region &region) {
  // Build function type matching the region signature.
  auto functionType = FunctionType::get(
      region.getContext(), region.getArgumentTypes(), /*results=*/{});

  // Clone region into the function body.
  auto funcOp = mlir::func::FuncOp::create(loc, functionName, functionType);
  IRMapping mapping;
  region.cloneInto(&funcOp.getFunctionBody(), mapping);

  // Replace flow.return with std.return.
  // NOTE: in the dispatch workgroups case the return should have no values.
  for (auto &block : funcOp.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(block.back())) {
      OpBuilder builder(returnOp);
      builder.create<mlir::func::ReturnOp>(
          returnOp.getLoc(), llvm::to_vector(returnOp.getOperands()));
      returnOp.erase();
    }
  }

  return funcOp;
}

// Outlines a dispatch region into a flow.executable and replaces the region op
// with a dispatch to that outlined executable.
static LogicalResult outlineDispatchWorkgroupsOp(
    std::string name, IREE::Flow::DispatchWorkgroupsOp dispatchWorkgroupsOp) {
  // Convert the region to a free-floating function.
  auto workgroupFuncOp =
      createWorkgroupFunc(dispatchWorkgroupsOp.getLoc(), name,
                          dispatchWorkgroupsOp.getWorkgroupBody());
  if (!workgroupFuncOp) {
    return failure();
  }

  // Create the executable with the region cloned into it.
  auto parentFuncOp =
      dispatchWorkgroupsOp->getParentOfType<mlir::FunctionOpInterface>();
  auto executableOp =
      createExecutable(dispatchWorkgroupsOp.getLoc(), name, {workgroupFuncOp},
                       parentFuncOp->getParentOfType<mlir::ModuleOp>());
  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Add an export pointing at the entry point function.
  OpBuilder builder(executableOp.getBody());
  auto exportOp = builder.create<IREE::Flow::ExecutableExportOp>(
      dispatchWorkgroupsOp.getLoc(), workgroupFuncOp.getName(),
      SymbolRefAttr::get(workgroupFuncOp));
  exportOp->setDialectAttrs(dispatchWorkgroupsOp->getDialectAttrs());

  // Move over the workgroup count region, if present.
  if (!dispatchWorkgroupsOp.getWorkgroupCount().empty()) {
    exportOp.getWorkgroupCount().takeBody(
        dispatchWorkgroupsOp.getWorkgroupCount());
  }

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertDispatchWorkgroupsToDispatchOp(dispatchWorkgroupsOp,
                                               executableOp, exportOp);
}

} // namespace

class OutlineDispatchRegionsPass
    : public OutlineDispatchRegionsBase<OutlineDispatchRegionsPass> {
public:
  OutlineDispatchRegionsPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    // Convert each dispatch region into a flow.executable + dispatch op.
    int initializerCount = 0;
    int funcLikeCount = 0;
    for (auto funcOp : getOperation().getOps<mlir::FunctionOpInterface>()) {
      // Generate a nice name if possible. All ops we outline in the same scope
      // will have the same root name.
      std::string namePrefix;
      if (isa<IREE::Util::InitializerOp>(funcOp)) {
        namePrefix =
            std::string("_initializer_") + std::to_string(initializerCount++);
      } else {
        namePrefix = funcOp.getName().str();
        if (namePrefix.empty()) {
          namePrefix =
              std::string("_func_like_") + std::to_string(funcLikeCount++);
        }
      }

      // Outline all of the dispatch regions ops in this function.
      SmallVector<Operation *> deadOps;
      auto outlineOps = [&](Operation *op) {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<IREE::Flow::DispatchWorkgroupsOp>(
                [&](auto dispatchWorkgroupsOp) {
                  if (failed(outlineDispatchWorkgroupsOp(
                          (namePrefix + "_dispatch_" +
                           llvm::Twine(deadOps.size()))
                              .str(),
                          dispatchWorkgroupsOp))) {
                    return WalkResult::interrupt();
                  }
                  deadOps.push_back(op);
                  return WalkResult::advance();
                })
            .Default(WalkResult::advance());
      };
      if (funcOp.walk(outlineOps).wasInterrupted())
        return signalPassFailure();
      for (auto *deadOp : deadOps)
        deadOp->erase();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchRegionsPass() {
  return std::make_unique<OutlineDispatchRegionsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
