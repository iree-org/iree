// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<int> clMaxAllocationSizeInBytes(
    "iree-llvmcpu-stack-allocation-limit",
    llvm::cl::desc("maximum allowed stack allocation size in bytes"),
    llvm::cl::init(32768));
static llvm::cl::opt<bool> clFailOnOutOfBoundsStackAllocation(
    "iree-llvmcpu-fail-on-out-of-bounds-stack-allocation",
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

/// Returns success if the cummulative stack allocation size is less than the
/// limit set by clMaxAllocationSizeInBytes.
static LogicalResult checkStackAllocationSize(func::FuncOp funcOp) {
  if (funcOp.getBody().empty()) return success();

  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk(
      [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  if (allocaOps.empty()) {
    return success();
  }

  int cumSize = 0;
  for (auto allocaOp : allocaOps) {
    if (allocaOp->getBlock() != &funcOp.getBody().front()) {
      return allocaOp->emitOpError(
          "all stack allocations need to be hoisted to the entry block of the "
          "function");
    }
    int allocaSize = 1;
    auto allocaType = llvm::cast<ShapedType>(allocaOp.getType());
    for (auto dimSize : allocaType.getShape()) {
      if (ShapedType::isDynamic(dimSize)) continue;
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
  auto moduleOp = getOperation();

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (clFailOnOutOfBoundsStackAllocation &&
        failed(checkStackAllocationSize(funcOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass() {
  return std::make_unique<LLVMCPUCheckIRBeforeLLVMConversionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
