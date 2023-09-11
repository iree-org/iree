// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/iree_reduce_lib.h"

#include "iree/compiler/Reducer/Strategies/DeltaStrategies.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace llvm;

Operation *
mlir::iree_compiler::ireeRunReducingStrategies(OwningOpRef<Operation *> module,
                                               StringRef testScript) {
  ModuleOp root = dyn_cast<ModuleOp>(module.release());
  WorkItem workItem(root);
  Oracle oracle(testScript);

  reduceFlowDispatchOperandToResultDelta(oracle, workItem);
  reduceFlowDispatchResultBySplatDelta(oracle, workItem);
  reduceLinalgOnTensorsDelta(oracle, workItem);

  return workItem.getModule();
}
