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

    // Collect calls to convert.
    SmallVector<IREE::VM::CallOp> callsToConvert;
    SmallVector<IREE::VM::CallVariadicOp> variadicCallsToConvert;
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
              variadicCallsToConvert.push_back(callVariadicOp);
            }
          });
    });

    // Convert each call.
    for (IREE::VM::CallOp callOp : callsToConvert) {
      if (failed(convertCallToYieldable(callOp))) {
        signalPassFailure();
        return;
      }
    }

    // Convert each variadic call.
    for (IREE::VM::CallVariadicOp callOp : variadicCallsToConvert) {
      if (failed(convertCallVariadicToYieldable(callOp))) {
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

  LogicalResult
  convertCallVariadicToYieldable(IREE::VM::CallVariadicOp callOp) {
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

    // Extract segment info.
    SmallVector<int16_t> segmentSizes;
    for (auto val : callOp.getSegmentSizes()) {
      segmentSizes.push_back(val.getSExtValue());
    }
    SmallVector<Type> segmentTypes;
    for (auto typeAttr : callOp.getSegmentTypes()) {
      segmentTypes.push_back(cast<TypeAttr>(typeAttr).getValue());
    }

    // Create the vm.call.variadic.yieldable op and erase the original call.
    builder.setInsertionPoint(callOp);
    IREE::VM::CallVariadicYieldableOp::create(
        builder, loc, builder.getAttr<FlatSymbolRefAttr>(callOp.getCallee()),
        segmentSizes, segmentTypes, callOp.getOperands(), resumeBlock,
        resultTypes);
    callOp.erase();

    return success();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::VM
