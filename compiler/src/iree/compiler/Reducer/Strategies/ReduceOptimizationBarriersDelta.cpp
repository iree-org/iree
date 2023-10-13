
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Strategies/DeltaStrategies.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::Reducer;

void mlir::iree_compiler::Reducer::reduceOptimizationBarriersDelta(
    ChunkManager &chunker, WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  SmallVector<IREE::Util::OptimizationBarrierOp> optBarriers;
  module.walk([&](IREE::Util::OptimizationBarrierOp optBarrier) {
    if (!chunker.shouldFeatureBeKept()) {
      optBarriers.push_back(optBarrier);
    }
  });

  if (optBarriers.empty()) {
    return;
  }

  // Replace all dispatch ops with the chosen operand.
  for (auto optBarrier : optBarriers) {
    optBarrier.replaceAllUsesWith(optBarrier.getOperands());
    optBarrier.erase();
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
