// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUASSIGNWORKGROUPLOCALMEMORYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

constexpr int64_t kMinWorkgroupLocalAlignment = 16;

struct LLVMCPUAssignWorkgroupLocalMemoryPass
    : impl::LLVMCPUAssignWorkgroupLocalMemoryPassBase<
          LLVMCPUAssignWorkgroupLocalMemoryPass> {
  using impl::LLVMCPUAssignWorkgroupLocalMemoryPassBase<
      LLVMCPUAssignWorkgroupLocalMemoryPass>::
      LLVMCPUAssignWorkgroupLocalMemoryPassBase;
  void runOnOperation() override;
};

static bool hasWorkgroupLocalMemorySpace(memref::AllocOp allocOp) {
  return isa_and_nonnull<IREE::Codegen::WorkgroupLocalMemoryAttr>(
      allocOp.getType().getMemorySpace());
}

static bool hasWorkgroupLocalMemorySpace(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  return memRefType && isa_and_nonnull<IREE::Codegen::WorkgroupLocalMemoryAttr>(
                           memRefType.getMemorySpace());
}

static LogicalResult checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (llvm::AddOverflow(lhs, rhs, result)) {
    return failure();
  }
  return success();
}

static LogicalResult checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (llvm::MulOverflow(lhs, rhs, result)) {
    return failure();
  }
  return success();
}

static FailureOr<int64_t>
computeStaticElementFootprint(memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  if (!type.hasStaticShape()) {
    allocOp.emitOpError(
        "workgroup local memory allocations must have static shape");
    return failure();
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    allocOp.emitOpError(
        "workgroup local memory allocations must have static layout");
    return failure();
  }
  if (ShapedType::isDynamic(offset) ||
      llvm::any_of(strides, ShapedType::isDynamic)) {
    allocOp.emitOpError(
        "workgroup local memory allocations must have static layout");
    return failure();
  }
  if (offset < 0 ||
      llvm::any_of(strides, [](int64_t stride) { return stride < 0; })) {
    allocOp.emitOpError(
        "workgroup local memory allocations must have non-negative layout");
    return failure();
  }

  if (llvm::is_contained(type.getShape(), 0)) {
    return 0;
  }

  int64_t maxElementOffset = offset;
  for (auto [dim, stride] : llvm::zip_equal(type.getShape(), strides)) {
    int64_t contribution = 0;
    if (failed(checkedMul(dim - 1, stride, contribution)) ||
        failed(checkedAdd(maxElementOffset, contribution, maxElementOffset))) {
      allocOp.emitOpError("workgroup local memory allocation size overflow");
      return failure();
    }
  }

  int64_t footprint = 0;
  if (failed(checkedAdd(maxElementOffset, 1, footprint))) {
    allocOp.emitOpError("workgroup local memory allocation size overflow");
    return failure();
  }
  return footprint;
}

static FailureOr<int64_t> computeAllocationByteSize(memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  FailureOr<int64_t> elementFootprint = computeStaticElementFootprint(allocOp);
  if (failed(elementFootprint)) {
    return failure();
  }

  DataLayout dataLayout = DataLayout::closest(allocOp);
  llvm::TypeSize elementByteSize =
      dataLayout.getTypeSize(type.getElementType());
  if (elementByteSize.isScalable()) {
    allocOp.emitOpError("workgroup local memory allocations must have "
                        "fixed-size element types");
    return failure();
  }
  if (elementByteSize.getFixedValue() >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    allocOp.emitOpError("workgroup local memory allocation size overflow");
    return failure();
  }
  int64_t elementBytes = static_cast<int64_t>(elementByteSize.getFixedValue());

  int64_t totalBytes = 0;
  if (failed(checkedMul(*elementFootprint, elementBytes, totalBytes))) {
    allocOp.emitOpError("workgroup local memory allocation size overflow");
    return failure();
  }
  return totalBytes;
}

static FailureOr<int64_t> computeAlignment(memref::AllocOp allocOp) {
  if (std::optional<uint64_t> explicitAlignment = allocOp.getAlignment()) {
    if (*explicitAlignment >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      allocOp.emitOpError(
          "workgroup local memory allocation alignment overflow");
      return failure();
    }
    return std::max(static_cast<int64_t>(*explicitAlignment),
                    kMinWorkgroupLocalAlignment);
  }
  DataLayout dataLayout = DataLayout::closest(allocOp);
  uint64_t abiAlignment =
      dataLayout.getTypeABIAlignment(allocOp.getType().getElementType());
  if (abiAlignment >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    allocOp.emitOpError("workgroup local memory allocation alignment overflow");
    return failure();
  }
  return std::max(static_cast<int64_t>(abiAlignment),
                  kMinWorkgroupLocalAlignment);
}

static LogicalResult
rejectUnsupportedWorkgroupLocalMemoryUses(mlir::FunctionOpInterface funcOp) {
  if (llvm::any_of(
          funcOp.getArgumentTypes(),
          [](Type type) { return hasWorkgroupLocalMemorySpace(type); }) ||
      llvm::any_of(funcOp.getResultTypes(), [](Type type) {
        return hasWorkgroupLocalMemorySpace(type);
      })) {
    return funcOp.emitOpError(
        "workgroup local memory is only supported for memref.alloc results");
  }

  WalkResult result = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
      if (hasWorkgroupLocalMemorySpace(allocaOp.getType())) {
        allocaOp.emitOpError(
            "workgroup local memory is only supported for memref.alloc");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace

void LLVMCPUAssignWorkgroupLocalMemoryPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  if (funcOp.getFunctionBody().empty()) {
    return;
  }
  if (failed(rejectUnsupportedWorkgroupLocalMemoryUses(funcOp))) {
    return signalPassFailure();
  }

  SmallVector<memref::AllocOp> localAllocs;
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (hasWorkgroupLocalMemorySpace(allocOp)) {
      localAllocs.push_back(allocOp);
    }
  });
  if (localAllocs.empty()) {
    return;
  }

  std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (!exportOp) {
    localAllocs.front().emitOpError(
        "workgroup local memory allocations are only supported in HAL "
        "executable exports");
    return signalPassFailure();
  }
  if (exportOp.value()->hasAttr(exportOp->getWorkgroupLocalMemoryAttrName())) {
    exportOp.value()->emitOpError(
        "already has a workgroup local memory requirement");
    return signalPassFailure();
  }

  OpBuilder builder(funcOp.getContext());
  int64_t currentOffset = 0;
  Block &entryBlock = funcOp.getFunctionBody().front();
  for (auto allocOp : localAllocs) {
    if (allocOp->getBlock() != &entryBlock) {
      allocOp.emitOpError(
          "workgroup local memory allocations must be in the function entry "
          "block");
      return signalPassFailure();
    }
    if (allocOp->hasAttr(kWorkgroupLocalMemoryRangeAttrName)) {
      allocOp.emitOpError("already has a workgroup local memory assignment");
      return signalPassFailure();
    }

    FailureOr<int64_t> byteSize = computeAllocationByteSize(allocOp);
    if (failed(byteSize)) {
      return signalPassFailure();
    }

    FailureOr<int64_t> alignment = computeAlignment(allocOp);
    if (failed(alignment)) {
      return signalPassFailure();
    }
    uint64_t alignedOffset = llvm::alignTo(static_cast<uint64_t>(currentOffset),
                                           static_cast<uint64_t>(*alignment));
    if (alignedOffset >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      allocOp.emitOpError("workgroup local memory allocation size overflow");
      return signalPassFailure();
    }
    currentOffset = static_cast<int64_t>(alignedOffset);
    allocOp->setAttr(kWorkgroupLocalMemoryRangeAttrName,
                     builder.getDenseI64ArrayAttr({currentOffset, *byteSize}));

    if (failed(checkedAdd(currentOffset, *byteSize, currentOffset))) {
      allocOp.emitOpError("workgroup local memory allocation size overflow");
      return signalPassFailure();
    }
  }

  exportOp.value()->setAttr(exportOp->getWorkgroupLocalMemoryAttrName(),
                            builder.getIndexAttr(currentOffset));
}

} // namespace mlir::iree_compiler
