// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUCHECKIRBEFORELLVMCONVERSIONPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
struct LLVMCPUCheckIRBeforeLLVMConversionPass
    : impl::LLVMCPUCheckIRBeforeLLVMConversionPassBase<
          LLVMCPUCheckIRBeforeLLVMConversionPass> {
  using impl::LLVMCPUCheckIRBeforeLLVMConversionPassBase<
      LLVMCPUCheckIRBeforeLLVMConversionPass>::
      LLVMCPUCheckIRBeforeLLVMConversionPassBase;
  void runOnOperation() override;
};
} // namespace

/// Returns success if the cumulative stack allocation size is less than the
/// limit set through --iree-llvmcpu-stack-allocation-limit (or the default
/// defined for HAL LLVMCPU target).
static LogicalResult
checkStackAllocationSize(mlir::FunctionOpInterface funcOp) {
  if (funcOp.getFunctionBody().empty()) {
    return success();
  }

  unsigned maxAllocationSizeInBytes = getStackAllocationLimit(funcOp);

  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk(
      [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  if (allocaOps.empty()) {
    return success();
  }

  int64_t cumSize = 0;
  for (auto allocaOp : allocaOps) {
    if (allocaOp->getBlock() != &funcOp.getFunctionBody().front()) {
      return allocaOp->emitOpError(
          "all stack allocations need to be hoisted to the entry block of the "
          "function");
    }
    FailureOr<int64_t> allocaSizeBytes = getStackAllocationSize(allocaOp);
    if (failed(allocaSizeBytes)) {
      return failure();
    }
    if (llvm::AddOverflow(cumSize, *allocaSizeBytes, cumSize)) {
      return allocaOp.emitOpError(
          "cumulative stack allocation size overflows 64 bits");
    }
  }
  if (cumSize > maxAllocationSizeInBytes) {
    return funcOp.emitOpError("exceeded stack allocation limit of ")
           << maxAllocationSizeInBytes << " bytes for function. Got " << cumSize
           << " bytes";
  }
  return success();
}

void LLVMCPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  if (!failOnOutOfBounds) {
    return;
  }

  mlir::FunctionOpInterface funcOp = getOperation();
  if (failed(checkStackAllocationSize(funcOp))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
