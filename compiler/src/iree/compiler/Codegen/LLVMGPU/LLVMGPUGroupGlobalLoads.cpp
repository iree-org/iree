// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-group-global-loads"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUGROUPGLOBALLOADSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Returns true if the operation is a load from global memory.
static bool isGlobalLoad(Operation *op) {
  Type memrefType;
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    memrefType = loadOp.getBase().getType();
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    memrefType = loadOp.getMemref().getType();
  } else {
    return false;
  }
  auto memref = dyn_cast<MemRefType>(memrefType);
  return memref && hasGlobalMemoryAddressSpace(memref);
}

/// Collects all ops in the same block that `op` transitively depends on
/// and that are strictly between `boundary` and `op`. These are the ops
/// that must be moved along with `op` if it is hoisted above `boundary`.
static void collectDepsInRange(Operation *op, Operation *boundary,
                               llvm::SetVector<Operation *> &deps) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != op->getBlock()) {
      continue;
    }
    if (!boundary->isBeforeInBlock(defOp)) {
      continue;
    }
    if (deps.insert(defOp)) {
      collectDepsInRange(defOp, boundary, deps);
    }
  }
}

/// Returns true if `op` can be moved before `boundary` while preserving SSA
/// dominance. `movedDeps` contains earlier dependencies that will also be
/// moved before `boundary`.
static bool
canMoveBeforeBoundary(Operation *op, Operation *boundary,
                      const llvm::SetVector<Operation *> &movedDeps) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != op->getBlock()) {
      continue;
    }
    if (defOp->isBeforeInBlock(boundary) || movedDeps.contains(defOp)) {
      continue;
    }
    return false;
  }
  return true;
}

/// Returns true if `op` writes to global memory. Used to decide
/// whether a non-dependent op left between the previous global load and the
/// load being hoisted would invalidate the load by changing observable
/// memory.
static bool writesToGlobalMemory(Operation *op) {
  if (isPure(op)) {
    return false;
  }
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp) {
    // Non-pure op without the memory-effects interface; conservatively assume
    // it could write to buffer or global memory.
    return true;
  }
  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }
    Value value = effect.getValue();
    if (!value) {
      // Write to an unknown resource; be conservative.
      return true;
    }
    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType) {
      continue;
    }
    if (hasGlobalMemoryAddressSpace(memrefType)) {
      return true;
    }
  }
  return false;
}

/// Groups global loads within each block by hoisting each load (along with
/// its pure address-computation dependencies) to be adjacent to the
/// preceding global load.
static void groupGlobalLoadsInBlock(Block &block) {
  SmallVector<Operation *> globalLoads;
  for (Operation &op : block) {
    if (isGlobalLoad(&op)) {
      globalLoads.push_back(&op);
    }
  }

  Operation *prevGlobalLoad = nullptr;
  for (Operation *load : globalLoads) {
    if (!prevGlobalLoad) {
      prevGlobalLoad = load;
      continue;
    }

    if (!prevGlobalLoad->isBeforeInBlock(load)) {
      prevGlobalLoad = load;
      continue;
    }

    // Collect the ops between `prevGlobalLoad` and `load` that the load
    // transitively depends on and would need to be hoisted alongside it.
    llvm::SetVector<Operation *> deps;
    collectDepsInRange(load, prevGlobalLoad, deps);

    // The dependencies move with the load, but they must be pure so that
    // hoisting them above unrelated ops in the range is safe.
    if (!llvm::all_of(deps, [](Operation *op) { return isPure(op); })) {
      prevGlobalLoad = load;
      continue;
    }

    // Non-dependent ops in (prevGlobalLoad, load) are left in place after the
    // load is hoisted. If any of them writes to global memory the
    // hoisted load could observe stale memory, so the move is unsafe.
    bool unsafe = false;
    for (Operation *cur = prevGlobalLoad->getNextNode(); cur && cur != load;
         cur = cur->getNextNode()) {
      if (deps.contains(cur)) {
        continue;
      }
      if (writesToGlobalMemory(cur)) {
        unsafe = true;
        break;
      }
    }
    if (unsafe) {
      prevGlobalLoad = load;
      continue;
    }

    // Move deps in topological order, then the load itself.
    // Since deps are collected via DFS, we need to sort them by their
    // original position in the block to maintain valid SSA ordering.
    SmallVector<Operation *> sortedDeps(deps.begin(), deps.end());
    llvm::sort(sortedDeps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    // If an address-computation dependency does not depend on the previous
    // global load, move it before that load. That lets the global loads become
    // adjacent while preserving the dependency order.
    llvm::SetVector<Operation *> depsBeforePrevLoad;
    SmallVector<Operation *> depsAfterPrevLoad;
    for (Operation *dep : sortedDeps) {
      if (canMoveBeforeBoundary(dep, prevGlobalLoad, depsBeforePrevLoad)) {
        depsBeforePrevLoad.insert(dep);
        continue;
      }
      depsAfterPrevLoad.push_back(dep);
    }

    for (Operation *dep : depsBeforePrevLoad) {
      dep->moveBefore(prevGlobalLoad);
    }

    Operation *insertAfter = prevGlobalLoad;
    for (Operation *dep : depsAfterPrevLoad) {
      dep->moveAfter(insertAfter);
      insertAfter = dep;
    }
    load->moveAfter(insertAfter);

    prevGlobalLoad = load;
  }
}

struct LLVMGPUGroupGlobalLoadsPass final
    : impl::LLVMGPUGroupGlobalLoadsPassBase<LLVMGPUGroupGlobalLoadsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    funcOp.walk([](Block *block) { groupGlobalLoadsInBlock(*block); });
  }
};

} // namespace

} // namespace mlir::iree_compiler
