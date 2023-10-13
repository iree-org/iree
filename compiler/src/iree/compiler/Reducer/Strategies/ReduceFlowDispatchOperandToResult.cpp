// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/Strategies/DeltaStrategies.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::Reducer;

void mlir::iree_compiler::Reducer::reduceFlowDispatchOperandToResultDelta(
    ChunkManager &chunker, WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  // Create an result to operand map.
  SmallVector<std::pair<Value, Value>> resultToOperand;
  module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    // Check if there is a result with the same type as the operand.
    for (auto result : dispatchOp.getResults()) {
      for (auto operand : dispatchOp.getOperands()) {
        if (operand.getType() == result.getType() &&
            !chunker.shouldFeatureBeKept()) {
          resultToOperand.push_back({result, operand});
          break;
        }
      }
    }
  });

  if (resultToOperand.empty()) {
    return;
  }

  // Replace all dispatch ops with the chosen operand.
  for (auto [result, operand] : resultToOperand) {
    result.replaceAllUsesWith(operand);
  }

  PassManager pm(module.getContext());
  // Dead code eliminate the dispatch ops.
  pm.addPass(createCSEPass());
  // Dead code eliminate the weights.
  pm.addPass(createSymbolDCEPass());
  // Canonicalize the module.
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(module))) {
    return;
  }
}
