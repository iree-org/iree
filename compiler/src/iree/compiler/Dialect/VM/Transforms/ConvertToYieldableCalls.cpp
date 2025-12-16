// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_CONVERTTOYIELDABLECALLSPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

namespace {

// Returns true if the callee is marked as yieldable (has vm.yield attribute).
static bool isCalleeYieldable(Operation *calleeOp) {
  return calleeOp && calleeOp->hasAttr("vm.yield");
}

class ConvertToYieldableCallsPass
    : public IREE::VM::impl::ConvertToYieldableCallsPassBase<
          ConvertToYieldableCallsPass> {
public:
  void runOnOperation() override {
    IREE::VM::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Collect calls to convert and variadic calls to error on.
    SmallVector<IREE::VM::CallOp> callsToConvert;
    SmallVector<IREE::VM::CallVariadicOp> variadicCallErrors;
    moduleOp.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<IREE::VM::CallOp>([&](auto callOp) {
            Operation *calleeOp = symbolTable.lookup(callOp.getCallee());
            if (isCalleeYieldable(calleeOp)) {
              callsToConvert.push_back(callOp);
            }
          })
          .Case<IREE::VM::CallVariadicOp>([&](auto callVariadicOp) {
            Operation *calleeOp =
                symbolTable.lookup(callVariadicOp.getCallee());
            if (isCalleeYieldable(calleeOp)) {
              variadicCallErrors.push_back(callVariadicOp);
            }
          });
    });

    // Error on variadic calls to yieldable functions (not supported).
    for (IREE::VM::CallVariadicOp callOp : variadicCallErrors) {
      callOp.emitError("vm.call.variadic to yieldable function not supported; "
                       "vm.call.yieldable does not support variadic arguments");
    }
    if (!variadicCallErrors.empty()) {
      return signalPassFailure();
    }

    // Convert each call.
    for (IREE::VM::CallOp callOp : callsToConvert) {
      if (failed(convertCallToYieldable(callOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult convertCallToYieldable(IREE::VM::CallOp callOp) {
    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    // Split block after the call to create resume block.
    Block *currentBlock = callOp->getBlock();
    Block *resumeBlock = currentBlock->splitBlock(callOp->getNextNode());

    // Add block arguments for call results.
    auto resultTypes = llvm::to_vector(callOp.getResultTypes());
    SmallVector<Location> argLocs(resultTypes.size(), loc);
    resumeBlock->addArguments(resultTypes, argLocs);

    // Replace uses of call results with block arguments.
    for (auto [result, blockArg] :
         llvm::zip_equal(callOp.getResults(), resumeBlock->getArguments())) {
      result.replaceAllUsesWith(blockArg);
    }

    // Create the vm.call.yieldable op and erase the original call.
    builder.setInsertionPoint(callOp);
    IREE::VM::CallYieldableOp::create(
        builder, loc, builder.getAttr<FlatSymbolRefAttr>(callOp.getCallee()),
        callOp.getOperands(), resumeBlock, resultTypes);
    callOp.erase();

    return success();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::VM
