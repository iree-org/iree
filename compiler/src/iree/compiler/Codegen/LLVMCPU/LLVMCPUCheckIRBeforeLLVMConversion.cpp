// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUCHECKIRBEFORELLVMCONVERSIONPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

static llvm::cl::opt<unsigned> clAssumedVscaleValue(
    "iree-llvmcpu-stack-allocation-assumed-vscale",
    llvm::cl::desc(
        "assumed value of vscale when checking (scalable) stack allocations"),
    llvm::cl::init(1));

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

/// Returns success if the cummulative stack allocation size is less than the
/// limit set by clMaxAllocationSizeInBytes.
static LogicalResult
checkStackAllocationSize(mlir::FunctionOpInterface funcOp) {
  if (funcOp.getFunctionBody().empty())
    return success();

  unsigned maxAllocationSizeInBytes = 32768;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (targetAttr) {
    auto nativeAllocationSizeAttr =
        getConfigIntegerAttr(targetAttr, "max_stack_allocation_size");
    if (nativeAllocationSizeAttr) {
      maxAllocationSizeInBytes = nativeAllocationSizeAttr->getInt();
    }
  }

  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk(
      [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  if (allocaOps.empty()) {
    return success();
  }

  int cumSize = 0;
  const unsigned assumedVscale = clAssumedVscaleValue;
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
      // Assume vscale is `clAssumedVscaleValue` for determining if the alloca
      // is within the stack limit. This should always resolve to a constant
      // bound. Note: This may be an underestimate if the runtime larger than
      // `clAssumedVscaleValue`, but should still catch unreasonable allocatons
      // (which will have large static factors).
      auto ub = vector::ScalableValueBoundsConstraintSet::computeScalableBound(
          operand, /*dim=*/std::nullopt,
          /*vscaleMin=*/assumedVscale,
          /*vscaleMax=*/assumedVscale, presburger::BoundType::UB);
      if (succeeded(ub)) {
        allocaSize *= ub->getSize()->baseSize;
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
  if (cumSize > maxAllocationSizeInBytes) {
    return funcOp.emitOpError("exceeded stack allocation limit of ")
           << maxAllocationSizeInBytes << " bytes for function. Got "
           << cumSize << " bytes";
  }
  return success();
}

void LLVMCPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  if (!failOnOutOfBounds) {
    return;
  }

  auto funcOp = getOperation();
  if (failed(checkStackAllocationSize(funcOp))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
