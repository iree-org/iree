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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::Reducer;

void mlir::iree_compiler::Reducer::reduceFlowDispatchResultBySplatDelta(
    ChunkManager &chunker, WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  // Create a list of dispatch ops we want to replace.
  SmallVector<IREE::Flow::DispatchOp> dispatchOps;
  SmallVector<IREE::Flow::DispatchOp> keepOps;
  module.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    bool shouldReplace = true;
    for (auto result : dispatchOp.getResults()) {
      // Only replace ranked tensor types.
      if (!isa<RankedTensorType>(result.getType())) {
        shouldReplace = false;
        break;
      }
    }

    if (shouldReplace && !chunker.shouldFeatureBeKept()) {
      dispatchOps.push_back(dispatchOp);
    } else {
      keepOps.push_back(dispatchOp);
    }
  });

  if (dispatchOps.empty()) {
    return;
  }

  OpBuilder builder(module.getContext());

  // Insert util.optimization_barrier for results of keep dispatch ops.
  for (auto dispatchOp : keepOps) {
    builder.setInsertionPointAfter(dispatchOp);
    for (Value result : dispatchOp.getResults()) {
      builder.create<IREE::Util::OptimizationBarrierOp>(dispatchOp.getLoc(),
                                                        result);
    }
  }

  // Replace all dispatch ops with flow.tensor.splat of 0.
  for (auto dispatch : dispatchOps) {
    builder.setInsertionPoint(dispatch);

    for (unsigned index = 0, e = dispatch->getNumResults(); index < e;
         ++index) {
      auto result = dispatch.getResults()[index];
      auto dynamicDims = dispatch.getResultDynamicDims(index);
      auto tensorType = result.getType().cast<RankedTensorType>();
      auto elType = tensorType.getElementType();
      auto zeroAttr = builder.getZeroAttr(elType);
      auto zero = builder.create<arith::ConstantOp>(result.getLoc(), zeroAttr);

      auto splat = builder.create<IREE::Flow::TensorSplatOp>(
          result.getLoc(), tensorType, zero, dynamicDims);
      result.replaceAllUsesWith(splat);
    }

    // Erase the dispatch.
    dispatch.erase();
  }

  PassManager pm(module.getContext());
  // Dead code eliminate the dispatch ops.
  pm.addPass(createCSEPass());
  // Dead code eliminate globals.
  pm.addPass(createSymbolDCEPass());
  // Canonicalize so that the splats are fused with reshapes.
  pm.addPass(createCanonicalizerPass());
  // CSE again to de-duplicate splats.
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    return;
  }
}
