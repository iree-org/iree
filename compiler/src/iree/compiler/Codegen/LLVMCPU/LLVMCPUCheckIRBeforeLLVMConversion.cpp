// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

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
      if (dimSize == ShapedType::kDynamicSize) continue;
      size *= dimSize;
    }
    for (auto operand : allocaOp.dynamicSizes()) {
      auto ub = linalg::getConstantUpperBoundForIndex(operand);
      if (failed(ub)) {
        return allocaOp.emitOpError(
            "expected no stack allocations without upper bound shapes");
      }
      size *= *ub;
    }
    size *= type.getElementType().getIntOrFloatBitWidth();
    if (allocaOp.alignment()) {
      int64_t alignmentInBits = *allocaOp.alignment() * 8;
      size = llvm::divideCeil(size, alignmentInBits) * alignmentInBits;
    }
    totalBits += size;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
  constexpr int k32KBInBits = 32 * 1024 * 8;
  if (totalBits > k32KBInBits) {
    moduleOp.emitOpError(
        "expected total size of stack allocation is not greater than 32 KB, "
        "but got ")
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
