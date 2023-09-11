// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <random>

#include "iree/compiler/Reducer/DeltaStratergies.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static void extractWeightsInModule(ChunkManager &chunker, WorkItem &workItem) {
  ModuleOp module = workItem.getModule();

  SmallVector<IREE::Util::GlobalOpInterface> globals;
  for (auto globalOp : module.getOps<IREE::Util::GlobalOpInterface>()) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(globalOp.getGlobalType())) {

      // Do not replace if it's already a byte pattern.
      if (isa<IREE::Util::BytePatternAttr>(globalOp.getGlobalInitialValue())) {
        continue;
      }

      if (!chunker.shouldFeatureBeKept())
        globals.push_back(globalOp);
    }
  }

  if (globals.empty()) {
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(0, INT64_MAX - 1);

  // Replace all globals with byte_pattern.
  OpBuilder builder = workItem.getBuilder();
  for (auto globalOp : globals) {
    // Replace this globalOp with a bytePattern.
    auto bytePattern = builder.getAttr<IREE::Util::BytePatternAttr>(
        globalOp.getGlobalType(), dis(gen));
    globalOp.setGlobalInitialValue(bytePattern);
  }
}

void mlir::iree_compiler::reduceWeightsToRandom(Oracle &oracle,
                                                WorkItem &workItem) {
  runDeltaPass(oracle, workItem, extractWeightsInModule,
               "Reducing Weights to Random Bytes");
}
