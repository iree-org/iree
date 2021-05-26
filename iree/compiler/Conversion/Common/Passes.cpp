// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Common/Passes.h"

#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void addLinalgBufferizePasses(OpPassManager &passManager,
                              WorkgroupMemoryAllocationFn allocationFn) {
  passManager.addNestedPass<FuncOp>(createLinalgBufferizePass(allocationFn));
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createBufferAllocViewCleanUpPass());
  // passManager.addPass(createBufferHoistingPass());
  // TODO(nicolasvasilache): bug in buffer loop hoisting with
  // dynamic_linalg_matmul_on_tensors_fuse_0.mlir
  // passManager.addPass(createBufferLoopHoistingPass());
}

}  // namespace iree_compiler
}  // namespace mlir
