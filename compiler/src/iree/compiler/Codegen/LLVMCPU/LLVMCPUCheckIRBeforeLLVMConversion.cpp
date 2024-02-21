// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

static llvm::cl::opt<int> clMaxAllocationSizeInBytes(
    "iree-llvmcpu-stack-allocation-limit",
    llvm::cl::desc("maximum allowed stack allocation size in bytes"),
    llvm::cl::init(32768));

namespace {
struct LLVMCPUCheckIRBeforeLLVMConversionPass
    : LLVMCPUCheckIRBeforeLLVMConversionBase<
          LLVMCPUCheckIRBeforeLLVMConversionPass> {
  LLVMCPUCheckIRBeforeLLVMConversionPass(bool failOnOutOfBounds) {
    this->failOnOutOfBounds = failOnOutOfBounds;
  }

  void runOnOperation() override;
};
} // namespace

/// Returns success if the cummulative stack allocation size is less than the
/// limit set by clMaxAllocationSizeInBytes.
static LogicalResult
checkStackAllocationSize(mlir::FunctionOpInterface funcOp) {
  if (funcOp.getFunctionBody().empty())
    return success();

  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk(
      [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  if (allocaOps.empty()) {
    return success();
  }

  int cumSize = 0;
  for (auto allocaOp : allocaOps) {
    if (allocaOp->getBlock() != &funcOp.getFunctionBody().front()) {
      return allocaOp->emitOpError(
          "all stack allocations need to be hoisted to the entry block of the "
          "function");
    }
    int allocaSize = 1;
    auto allocaType = llvm::cast<ShapedType>(allocaOp.getType());
    for (auto dimSize : allocaType.getShape()) {
      if (ShapedType::isDynamic(dimSize))
        continue;
      allocaSize *= dimSize;
    }
    for (auto operand : allocaOp.getDynamicSizes()) {
      auto ub = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, /*dim=*/std::nullopt,
          /*stopCondition=*/nullptr, /*closedUB=*/true);
      if (succeeded(ub)) {
        allocaSize *= ub.value();
        continue;
      }
      return allocaOp.emitOpError("expected no unbounded stack allocations");
    }
    allocaSize *= allocaType.getElementType().getIntOrFloatBitWidth();
    if (allocaOp.getAlignment()) {
      int64_t alignmentInBits = *allocaOp.getAlignment() * 8;
      allocaSize =
          (llvm::divideCeil(allocaSize, alignmentInBits) * alignmentInBits);
    }
    cumSize += allocaSize / 8;
  }
  if (cumSize > clMaxAllocationSizeInBytes) {
    return funcOp.emitOpError("exceeded stack allocation limit of ")
           << clMaxAllocationSizeInBytes.getValue()
           << " bytes for function. Got " << cumSize << " bytes";
  }
  return success();
}

void LLVMCPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  if (!failOnOutOfBounds) {
    return;
  }

  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    if (failed(checkStackAllocationSize(funcOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass(bool failOnOutOfBounds) {
  return std::make_unique<LLVMCPUCheckIRBeforeLLVMConversionPass>(
      failOnOutOfBounds);
}

} // namespace mlir::iree_compiler
