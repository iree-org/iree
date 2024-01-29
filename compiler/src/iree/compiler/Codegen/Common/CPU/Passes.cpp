// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
//===---------------------------------------------------------------------===//

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  auto funcOp =
      builder.getInsertionPoint()->getParentOfType<mlir::FunctionOpInterface>();
  if (funcOp) {
    std::optional<Value> hoistedAllocation =
        hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
            funcOp, builder, loc, memRefType, dynamicSizes, alignment);
    if (hoistedAllocation) {
      return hoistedAllocation.value();
    }
  }
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  createLinalgCopyOp(builder, loc, from, to);
  return success();
}

void addCPUBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn = cpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, memcpyFn);
}

//===---------------------------------------------------------------------===//
// Register Common/CPU Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"
} // namespace

void registerCodegenCommonCPUPasses() {
  // Generated.
  registerPasses();
}
} // namespace mlir::iree_compiler
