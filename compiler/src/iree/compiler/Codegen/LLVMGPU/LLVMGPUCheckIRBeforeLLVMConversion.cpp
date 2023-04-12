// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<int> clMaxGPUSharedMemSize(
    "iree-llvmgpu-shared-mem-allocation-limit",
    llvm::cl::desc("maximum allowed shared memory size in bytes"),
    llvm::cl::init(163 * 1024));

namespace {
struct LLVMGPUCheckIRBeforeLLVMConversionPass
    : LLVMGPUCheckIRBeforeLLVMConversionBase<
          LLVMGPUCheckIRBeforeLLVMConversionPass> {
  void runOnOperation() override;
};
}  // namespace

static int shapedTypeStaticSize(ShapedType shapedType) {
  int allocSize = 1;
  for (auto dimSize : shapedType.getShape()) {
    if (ShapedType::isDynamic(dimSize)) continue;
    allocSize *= dimSize;
  }
  if (auto elementType = shapedType.getElementType().dyn_cast<ShapedType>()) {
    allocSize *= shapedTypeStaticSize(elementType);
  } else {
    allocSize *= shapedType.getElementType().getIntOrFloatBitWidth();
  }
  return allocSize;
}

/// Returns success if the total shared memory allocation size is less than the
/// limit set by clMaxGPUSharedMemSize.
static LogicalResult checkGPUAllocationSize(func::FuncOp funcOp) {
  if (funcOp.getBody().empty()) return success();

  SmallVector<memref::AllocOp> allocOps;
  funcOp.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });
  if (allocOps.empty()) {
    return success();
  }

  int cumSize = 0;
  for (auto allocOp : allocOps) {
    auto allocType = allocOp.getType().cast<MemRefType>();
    if (!hasSharedMemoryAddressSpace(allocType)) {
      continue;
    }
    if (!allocOp.getDynamicSizes().empty()) {
      return allocOp.emitOpError(
          "dynamic shared memory allocations unsupported.");
    }
    int allocSize = shapedTypeStaticSize(allocType);
    if (allocOp.getAlignment()) {
      int64_t alignmentInBits = *allocOp.getAlignment() * 8;
      allocSize =
          (llvm::divideCeil(allocSize, alignmentInBits) * alignmentInBits);
    }
    cumSize += allocSize / 8;
  }
  if (cumSize > clMaxGPUSharedMemSize) {
    return funcOp.emitOpError("exceeded GPU memory limit of ")
           << clMaxGPUSharedMemSize.getValue() << " bytes for function. Got "
           << cumSize << " bytes";
  }
  return success();
}

void LLVMGPUCheckIRBeforeLLVMConversionPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (failed(checkGPUAllocationSize(funcOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUCheckIRBeforeLLVMConversionPass() {
  return std::make_unique<LLVMGPUCheckIRBeforeLLVMConversionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
