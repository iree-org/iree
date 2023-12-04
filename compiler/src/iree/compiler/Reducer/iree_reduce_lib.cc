// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"

#include "iree/compiler/Reducer/Strategies/DeltaStrategies.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace llvm;

Operation *mlir::iree_compiler::Reducer::ireeRunReducingStrategies(
    OwningOpRef<Operation *> module, ReducerConfig &config) {
  ModuleOp root = dyn_cast<ModuleOp>(module.release());
  WorkItem workItem(root);
  Oracle oracle(config.testScript, config.useBytecode);
  Delta delta(oracle, workItem);

  // Check if the initial module is interesting.
  if (!oracle.isInteresting(workItem)) {
    llvm::report_fatal_error(
        "Initial test input is not interesting. Make sure you "
        "are returning 0 in your interesting check script if the input is "
        "interesting. Try running your interesting check script on the input "
        "and check the return code.");
  }

  delta.runDeltaPass(reduceOperandToResultDelta,
                     "Dispatch operand to result delta");
  delta.runDeltaPass(reduceFlowDispatchResultBySplatDelta,
                     "Dispatch result to splat delta");
  delta.runDeltaPass(reduceOptimizationBarriersDelta,
                     "Optimization barriers delta");
  delta.runDeltaPass(reduceLinalgOnTensorsDelta, "Linalg on tensors delta");
  delta.runDeltaPass(reduceOptimizationBarriersDelta,
                     "Optimization barriers delta");

  return workItem.getModule();
}
