// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/Strategies/DeltaStrategies.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static void extractLinalgOpsInModule(ChunkManager &chunker,
                                     WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  SmallVector<linalg::LinalgOp> linalgOps;
  module.walk([&](linalg::LinalgOp op) {
    if (!op.hasTensorSemantics())
      return;
    // Op should have at least on tensor input, otherwise it is not worth
    // reducing.
    bool hasAtleastOneTensorInput = false;
    for (auto *input : op.getDpsInputOperands()) {
      if (isa<RankedTensorType>(input->get().getType())) {
        hasAtleastOneTensorInput = true;
        break;
      }
    }

    if (!hasAtleastOneTensorInput)
      return;

    // There should be only 1 tensor output.
    if (op.getNumDpsInits() != 1)
      return;
    if (!isa<RankedTensorType>(op.getDpsInitOperand(0)->get().getType()))
      return;

    if (!chunker.shouldFeatureBeKept()) {
      linalgOps.push_back(op);
    }
  });

  if (linalgOps.empty()) {
    return;
  }

  OpBuilder builder = workItem.getBuilder();
  for (auto linalgOp : linalgOps) {
    builder.setInsertionPoint(linalgOp);
    Value out = linalgOp->getResult(0);
    Value init = linalgOp.getDpsInitOperand(0)->get();
    RankedTensorType outType = cast<RankedTensorType>(out.getType());
    Value newOut;

    // TODO: Add support for dynamic shapes using ValueBoundsInterface.
    if (outType.hasStaticShape()) {
      for (auto *input : linalgOp.getDpsInputOperands()) {
        auto inType = dyn_cast<RankedTensorType>(input->get().getType());
        if (!inType)
          continue;

        // Check if we can replace an input directly with the output.
        if (inType == outType) {
          newOut = input->get();
          break;
        }
        // TODO: Add support for different shapes using extract_slice/broadcast.
      }
    }

    if (newOut) {
      out.replaceAllUsesWith(newOut);
      linalgOp->erase();
      continue;
    }

    Type elType = outType.getElementType();
    // Build a constant 0 of the type.
    builder.setInsertionPoint(linalgOp);
    auto zero = builder
                    .create<arith::ConstantOp>(linalgOp.getLoc(),
                                               builder.getZeroAttr(elType))
                    .getResult();

    // Build linalg.fill for this out.
    newOut = builder.create<linalg::FillOp>(linalgOp.getLoc(), zero, init)
                 .getResult(0);

    out.replaceAllUsesWith(newOut);
    linalgOp->erase();
  }

  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module)))
    return;
}

void mlir::iree_compiler::reduceLinalgOnTensorsDelta(Oracle &oracle,
                                                     WorkItem &workItem) {
  runDeltaPass(oracle, workItem, extractLinalgOpsInModule,
               "Reducing Linalg ops");
}
