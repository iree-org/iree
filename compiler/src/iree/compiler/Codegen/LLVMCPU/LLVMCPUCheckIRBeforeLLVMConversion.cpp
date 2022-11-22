// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<int> clMaxAllocationSizeInBytes(
    "iree-llvmcpu-stack-allocation-limit",
    llvm::cl::desc("maximum allowed stack allocation size in bytes"),
    llvm::cl::init(32768));
static llvm::cl::opt<bool> clFailUnboundDynamicStackAllocation(
    "iree-llvmcpu-fail-unbound-dynamic-stack-allocation",
    llvm::cl::desc("fail if the upper bound of dynamic stack allocation cannot "
                   "be solved"),
    llvm::cl::init(true));

namespace {
struct LLVMCPUCheckIRBeforeLLVMConversionPass
    : LLVMCPUCheckIRBeforeLLVMConversionBase<
          LLVMCPUCheckIRBeforeLLVMConversionPass> {
  void runOnOperation() override;
};
}  // namespace

void LLVMCPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  auto moduleOp = getOperation();
  int64_t totalBits = 0;
  auto walkResult = moduleOp.walk([&](memref::AllocaOp allocaOp) -> WalkResult {
    auto type = allocaOp.getType().cast<ShapedType>();
    int64_t size = 1;
    for (auto dimSize : type.getShape()) {
      if (dimSize == ShapedType::kDynamic) continue;
      size *= dimSize;
    }
    for (auto operand : allocaOp.getDynamicSizes()) {
      auto ub = linalg::getConstantUpperBoundForIndex(operand);
      if (succeeded(ub)) {
        size *= *ub;
      } else if (clFailUnboundDynamicStackAllocation) {
        return allocaOp.emitOpError(
            "expected no stack allocations without upper bound shapes");
      }
    }
    size *= type.getElementType().getIntOrFloatBitWidth();
    if (allocaOp.getAlignment()) {
      int64_t alignmentInBits = *allocaOp.getAlignment() * 8;
      size = llvm::divideCeil(size, alignmentInBits) * alignmentInBits;
    }
    totalBits += size;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
  int maxAllocationSizeInBits = clMaxAllocationSizeInBytes * 8;
  if (totalBits > maxAllocationSizeInBits) {
    moduleOp.emitOpError(
        "expected total size of stack allocation is not greater than ")
        << clMaxAllocationSizeInBytes.getValue() << " bytes, but got "
        << llvm::divideCeil(totalBits, 8) << " bytes";
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass() {
  return std::make_unique<LLVMCPUCheckIRBeforeLLVMConversionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
