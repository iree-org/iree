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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  using Base::Base;
  void runOnOperation() override;
};

struct LocalAllocLayout {
  memref::AllocOp allocOp;
  int64_t byteOffset = 0;
  int64_t elementFootprint = 0;
};

struct AllocationSize {
  int64_t elementFootprint = 0;
  int64_t byteSize = 0;
};

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

static LogicalResult checkedAlignTo(int64_t value, int64_t alignment,
                                    int64_t &result) {
  assert(value >= 0);
  assert(alignment > 0);
  int64_t remainder = value % alignment;
  if (remainder == 0) {
    result = value;
    return success();
  }
  return checkedAdd(value, alignment - remainder, result);
}

static bool hasZeroElementShape(MemRefType type) {
  return llvm::is_contained(type.getShape(), 0);
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

  if (hasZeroElementShape(type)) {
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

static FailureOr<AllocationSize>
computeAllocationSize(memref::AllocOp allocOp) {
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
  AllocationSize allocationSize;
  allocationSize.elementFootprint = *elementFootprint;
  allocationSize.byteSize = totalBytes;
  return allocationSize;
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
collectWorkgroupLocalAllocs(mlir::FunctionOpInterface funcOp,
                            SmallVectorImpl<memref::AllocOp> &localAllocs) {
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
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        if (llvm::any_of(block.getArgumentTypes(), [](Type type) {
              return hasWorkgroupLocalMemorySpace(type);
            })) {
          op->emitOpError(
              "workgroup local memory is only supported for memref.alloc "
              "results");
          return WalkResult::interrupt();
        }
      }
    }

    for (Value result : op->getResults()) {
      if (!hasWorkgroupLocalMemorySpace(result.getType())) {
        continue;
      }
      auto allocOp = dyn_cast<memref::AllocOp>(op);
      if (!allocOp || result != allocOp.getMemref()) {
        op->emitOpError(
            "workgroup local memory is only supported for memref.alloc "
            "results");
        return WalkResult::interrupt();
      }
      localAllocs.push_back(allocOp);
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

  SmallVector<memref::AllocOp> localAllocs;
  if (failed(collectWorkgroupLocalAllocs(funcOp, localAllocs))) {
    return signalPassFailure();
  }
  if (localAllocs.empty()) {
    return;
  }

  IREE::Codegen::DispatchConfigOp configOp = getDispatchConfigOp(funcOp);
  if (!configOp) {
    localAllocs.front().emitOpError(
        "workgroup local memory allocations require an "
        "iree_codegen.dispatch_config op");
    return signalPassFailure();
  }
  if (configOp.getWorkgroupLocalMemoryAttr()) {
    configOp.emitOpError("already has a workgroup local memory requirement");
    return signalPassFailure();
  }

  int64_t currentOffset = 0;
  SmallVector<LocalAllocLayout> layouts;
  layouts.reserve(localAllocs.size());
  Block &entryBlock = funcOp.getFunctionBody().front();
  for (auto allocOp : localAllocs) {
    if (allocOp->getBlock() != &entryBlock) {
      allocOp.emitOpError(
          "workgroup local memory allocations must be in the function entry "
          "block");
      return signalPassFailure();
    }

    FailureOr<AllocationSize> allocationSize = computeAllocationSize(allocOp);
    if (failed(allocationSize)) {
      return signalPassFailure();
    }

    FailureOr<int64_t> alignment = computeAlignment(allocOp);
    if (failed(alignment)) {
      return signalPassFailure();
    }
    if (failed(checkedAlignTo(currentOffset, *alignment, currentOffset))) {
      allocOp.emitOpError("workgroup local memory allocation size overflow");
      return signalPassFailure();
    }
    layouts.push_back(LocalAllocLayout{
        /*allocOp=*/allocOp,
        /*byteOffset=*/currentOffset,
        /*elementFootprint=*/allocationSize->elementFootprint,
    });

    if (failed(checkedAdd(currentOffset, allocationSize->byteSize,
                          currentOffset))) {
      allocOp.emitOpError("workgroup local memory allocation size overflow");
      return signalPassFailure();
    }
  }

  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entryBlock);
  Attribute memorySpace = localAllocs.front().getType().getMemorySpace();
  MemRefType packedType = MemRefType::get({currentOffset}, builder.getI8Type(),
                                          AffineMap(), memorySpace);
  Value packedAlloc =
      memref::AllocOp::create(builder, funcOp.getLoc(), packedType);
  for (const LocalAllocLayout &layout : layouts) {
    memref::AllocOp allocOp = layout.allocOp;
    MemRefType allocType = allocOp.getType();
    MemRefType viewType = allocType;
    if (!allocType.getLayout().isIdentity()) {
      viewType =
          MemRefType::get({layout.elementFootprint}, allocType.getElementType(),
                          AffineMap(), allocType.getMemorySpace());
    }
    builder.setInsertionPoint(allocOp);
    Value byteOffset = arith::ConstantIndexOp::create(builder, allocOp.getLoc(),
                                                      layout.byteOffset);
    Value view = memref::ViewOp::create(builder, allocOp.getLoc(), viewType,
                                        packedAlloc, byteOffset, ValueRange{});
    if (viewType != allocType) {
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      if (failed(allocType.getStridesAndOffset(strides, offset))) {
        allocOp.emitOpError(
            "workgroup local memory allocations must have static layout");
        return signalPassFailure();
      }
      view = memref::ReinterpretCastOp::create(builder, allocOp.getLoc(),
                                               allocType, view, offset,
                                               allocType.getShape(), strides);
    }
    allocOp.replaceAllUsesWith(view);
    allocOp.erase();
  }

  configOp.setWorkgroupLocalMemoryAttr(builder.getIndexAttr(currentOffset));
}

} // namespace mlir::iree_compiler
