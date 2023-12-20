// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
class GPUCheckResourceUsagePass final
    : public GPUCheckResourceUsageBase<GPUCheckResourceUsagePass> {
public:
  explicit GPUCheckResourceUsagePass(
      std::function<unsigned(func::FuncOp)> getSharedMemoryLimit,
      std::function<unsigned(func::FuncOp)> getIndexBitwidth)
      : getSharedMemoryLimit(getSharedMemoryLimit),
        getIndexBitwidth(getIndexBitwidth) {}

  void runOnOperation() override;

private:
  std::function<unsigned(func::FuncOp)> getSharedMemoryLimit;
  std::function<unsigned(func::FuncOp)> getIndexBitwidth;
};
} // namespace

static unsigned getDatalayoutIndexBitwidth(func::FuncOp func) {
  auto mod = func->getParentOfType<ModuleOp>();
  LowerToLLVMOptions options(mod.getContext(), DataLayout(mod));
  return options.getIndexBitwidth();
}

static int
shapedTypeStaticSize(memref::AllocOp allocOp, ShapedType shapedType,
                     std::function<unsigned(func::FuncOp)> getIndexBitwidth) {
  int allocSize = 1;
  for (auto dimSize : shapedType.getShape()) {
    if (ShapedType::isDynamic(dimSize))
      continue;
    allocSize *= dimSize;
  }
  if (auto elementType =
          llvm::dyn_cast<ShapedType>(shapedType.getElementType())) {
    allocSize *= shapedTypeStaticSize(allocOp, elementType, getIndexBitwidth);
  } else {
    auto eltTy = shapedType.getElementType();
    if (eltTy.isIndex()) {
      auto func = allocOp->getParentOfType<func::FuncOp>();
      assert(getIndexBitwidth &&
             "getIndexBitwidth should have been set earlier");
      allocSize *= getIndexBitwidth(func);
    } else
      allocSize *= IREE::Util::getTypeBitWidth(shapedType.getElementType());
  }
  return allocSize;
}

/// Returns success if the total shared memory allocation size is less than the
/// limit set by limit.
static LogicalResult
checkGPUAllocationSize(func::FuncOp funcOp, unsigned limit,
                       std::function<unsigned(func::FuncOp)> getIndexBitwidth) {
  if (funcOp.getBody().empty())
    return success();

  SmallVector<memref::AllocOp> allocOps;
  funcOp.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });
  if (allocOps.empty())
    return success();

  int cumSize = 0;
  for (auto allocOp : allocOps) {
    auto allocType = llvm::cast<MemRefType>(allocOp.getType());
    if (!hasSharedMemoryAddressSpace(allocType))
      continue;

    if (!allocOp.getDynamicSizes().empty()) {
      return allocOp.emitOpError(
          "has unsupported dynamic shared memory allocations");
    }

    int allocSize = shapedTypeStaticSize(allocOp, allocType, getIndexBitwidth);
    if (allocOp.getAlignment()) {
      int64_t alignmentInBits = *allocOp.getAlignment() * 8;
      allocSize =
          (llvm::divideCeil(allocSize, alignmentInBits) * alignmentInBits);
    }
    cumSize += allocSize / 8;
  }
  if (cumSize > limit) {
    return funcOp.emitOpError("uses ")
           << cumSize << " bytes of shared memory; exceeded the limit of "
           << limit << " bytes";
  }
  return success();
}

void GPUCheckResourceUsagePass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    unsigned limit = this->getSharedMemoryLimit
                         ? this->getSharedMemoryLimit(funcOp)
                         : 64 * 1024;
    if (failed(checkGPUAllocationSize(funcOp, limit,
                                      this->getIndexBitwidth
                                          ? this->getIndexBitwidth
                                          : getDatalayoutIndexBitwidth))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createGPUCheckResourceUsagePass(
    std::function<unsigned(func::FuncOp)> getSharedMemoryLimit,
    std::function<unsigned(func::FuncOp)> getIndexBitwidth) {
  return std::make_unique<GPUCheckResourceUsagePass>(getSharedMemoryLimit,
                                                     getIndexBitwidth);
}

} // namespace mlir::iree_compiler
