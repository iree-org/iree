// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUCHECKRESOURCEUSAGEPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
static unsigned getDatalayoutIndexBitwidth(mlir::FunctionOpInterface func) {
  auto mod = func->getParentOfType<ModuleOp>();
  LowerToLLVMOptions options(mod.getContext(), DataLayout(mod));
  return options.getIndexBitwidth();
}

template <typename AllocTy>
static int shapedTypeStaticSize(
    AllocTy allocOp, ShapedType shapedType,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
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
      auto func =
          allocOp->template getParentOfType<mlir::FunctionOpInterface>();
      assert(getIndexBitwidth &&
             "getIndexBitwidth should have been set earlier");
      allocSize *= getIndexBitwidth(func);
    } else
      allocSize *= IREE::Util::getTypeBitWidth(shapedType.getElementType());
  }
  return allocSize;
}

template <typename AllocTy>
static FailureOr<int64_t> getAllocSize(
    AllocTy allocOp,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  auto allocType = llvm::cast<MemRefType>(allocOp.getType());
  if (!hasSharedMemoryAddressSpace(allocType)) {
    return 0;
  }

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
  return allocSize;
}

/// Returns success if the total shared memory allocation size is less than the
/// limit.
static LogicalResult checkGPUAllocationSize(
    mlir::FunctionOpInterface funcOp, unsigned limit,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  if (funcOp.getFunctionBody().empty())
    return success();

  SmallVector<Operation *> allocOps;
  funcOp.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });
  funcOp.walk([&](memref::AllocaOp allocOp) { allocOps.push_back(allocOp); });
  if (allocOps.empty())
    return success();

  int cumSize = 0;
  for (auto op : allocOps) {
    FailureOr<int64_t> allocSize = failure();
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocSize = getAllocSize(allocOp, getIndexBitwidth);
    } else if (auto allocOp = dyn_cast<memref::AllocaOp>(op)) {
      allocSize = getAllocSize(allocOp, getIndexBitwidth);
    }
    if (failed(allocSize)) {
      return failure();
    }
    cumSize += allocSize.value() / 8;
  }
  if (cumSize > limit) {
    return funcOp.emitOpError("uses ")
           << cumSize << " bytes of shared memory; exceeded the limit of "
           << limit << " bytes";
  }
  return success();
}

class GPUCheckResourceUsagePass final
    : public impl::GPUCheckResourceUsagePassBase<GPUCheckResourceUsagePass> {
public:
  explicit GPUCheckResourceUsagePass(
      std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth)
      : getIndexBitwidth(getIndexBitwidth) {}

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    unsigned limit =
        target ? target.getWgp().getMaxWorkgroupMemoryBytes() : 64 * 1024;
    if (failed(checkGPUAllocationSize(funcOp, limit,
                                      getIndexBitwidth
                                          ? getIndexBitwidth
                                          : getDatalayoutIndexBitwidth))) {
      return signalPassFailure();
    }
  }

private:
  std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGPUCheckResourceUsagePass(
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth) {
  return std::make_unique<GPUCheckResourceUsagePass>(getIndexBitwidth);
}

} // namespace mlir::iree_compiler
