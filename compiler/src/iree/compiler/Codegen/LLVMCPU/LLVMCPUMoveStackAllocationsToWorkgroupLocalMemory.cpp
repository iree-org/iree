// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUMOVESTACKALLOCATIONSTOWORKGROUPLOCALMEMORYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
struct LLVMCPUMoveStackAllocationsToWorkgroupLocalMemoryPass
    : impl::LLVMCPUMoveStackAllocationsToWorkgroupLocalMemoryPassBase<
          LLVMCPUMoveStackAllocationsToWorkgroupLocalMemoryPass> {
  using Base::Base;
  void runOnOperation() override;
};

struct StackAllocation {
  memref::AllocaOp allocaOp;
  int64_t byteSize = 0;
};
} // namespace

/// Returns the number of bytes the stack allocation check accounts for
/// `allocaOp`, or std::nullopt if the allocation cannot be moved: only
/// statically shaped allocations in the default memory space with an identity
/// layout are candidates.
static std::optional<int64_t> getMovableAllocationSize(memref::AllocaOp op) {
  MemRefType type = op.getType();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      type.getMemorySpace()) {
    return std::nullopt;
  }
  FailureOr<int64_t> sizeInBits =
      getStaticShapeSizeInBits(type, [](Type elementType) -> int64_t {
        return IREE::Util::getTypeBitWidth(elementType);
      });
  if (failed(sizeInBits) || *sizeInBits <= 0) {
    return std::nullopt;
  }
  int64_t size = *sizeInBits;
  if (std::optional<uint64_t> alignment = op.getAlignment()) {
    int64_t alignmentInBits = *alignment * 8;
    size = llvm::divideCeil(size, alignmentInBits) * alignmentInBits;
  }
  return llvm::divideCeil(size, 8);
}

void LLVMCPUMoveStackAllocationsToWorkgroupLocalMemoryPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  if (funcOp.getFunctionBody().empty()) {
    return;
  }

  // Same limit as the stack allocation check that runs after this pass.
  int64_t maxStackAllocationSize = 32 * 1024;
  if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    if (std::optional<int64_t> configuredSize =
            getConfigMaxStackAllocationSize(targetAttr.getConfiguration())) {
      maxStackAllocationSize = *configuredSize;
    }
  }

  // Only entry block allocations are accepted by the stack allocation check
  // and by the workgroup local memory assignment.
  Block &entryBlock = funcOp.getFunctionBody().front();
  SmallVector<StackAllocation> candidates;
  int64_t stackSize = 0;
  for (auto allocaOp : entryBlock.getOps<memref::AllocaOp>()) {
    std::optional<int64_t> byteSize = getMovableAllocationSize(allocaOp);
    if (!byteSize) {
      continue;
    }
    candidates.push_back(StackAllocation{allocaOp, *byteSize});
    stackSize += *byteSize;
  }
  if (stackSize <= maxStackAllocationSize) {
    return;
  }

  // The workgroup local memory requirement is recorded on the dispatch config
  // and must not have been set by someone else.
  IREE::Codegen::DispatchConfigOp configOp = getDispatchConfigOp(funcOp);
  if (!configOp || configOp.getWorkgroupLocalMemoryAttr()) {
    return;
  }

  // Move the largest allocations first until the rest fits on the stack. This
  // keeps the small, frequently accessed temporaries on the stack.
  llvm::stable_sort(candidates,
                    [](const StackAllocation &lhs, const StackAllocation &rhs) {
                      return lhs.byteSize > rhs.byteSize;
                    });
  Attribute memorySpace =
      IREE::Codegen::WorkgroupLocalMemoryAttr::get(funcOp.getContext());
  for (const StackAllocation &candidate : candidates) {
    if (stackSize <= maxStackAllocationSize) {
      break;
    }
    memref::AllocaOp allocaOp = candidate.allocaOp;
    MemRefType type = allocaOp.getType();
    MemRefType localType = MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(), memorySpace);
    OpBuilder builder(allocaOp);
    auto allocOp =
        memref::AllocOp::create(builder, allocaOp.getLoc(), localType,
                                ValueRange{}, allocaOp.getAlignmentAttr());
    // Users keep seeing a memref in the default memory space.
    Value replacement = memref::MemorySpaceCastOp::create(
        builder, allocaOp.getLoc(), type, allocOp.getMemref());
    allocaOp.replaceAllUsesWith(replacement);
    allocaOp.erase();
    stackSize -= candidate.byteSize;
  }
}

} // namespace mlir::iree_compiler
