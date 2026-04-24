// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "iree-codegen-rocdl-group-buffer-loads"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLGROUPBUFFERLOADSPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

/// Returns true if the operation is a vector.load from a fat raw buffer.
static bool isBufferLoad(Operation *op) {
  auto loadOp = dyn_cast<vector::LoadOp>(op);
  if (!loadOp) {
    return false;
  }
  auto memrefType = dyn_cast<MemRefType>(loadOp.getBase().getType());
  return memrefType && hasAMDGPUFatRawBufferAddressSpace(memrefType);
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
    if (!boundary->isBeforeInBlock(defOp) || !defOp->isBeforeInBlock(op)) {
      continue;
    }
    if (deps.insert(defOp)) {
      collectDepsInRange(defOp, boundary, deps);
    }
  }
}

/// Returns true if any op in `deps` has a side effect or is a load/store
/// that could alias with other memory operations between the boundary and
/// the load. We conservatively bail out if any dep has memory effects.
static bool hasMemoryEffects(const llvm::SetVector<Operation *> &deps) {
  for (Operation *op : deps) {
    if (!isPure(op)) {
      return true;
    }
  }
  return false;
}

/// Groups buffer loads within each block by hoisting each load (along with
/// its pure address-computation dependencies) to be adjacent to the
/// preceding buffer load.
static void groupBufferLoadsInBlock(Block &block) {
  SmallVector<Operation *> bufferLoads;
  for (Operation &op : block) {
    if (isBufferLoad(&op)) {
      bufferLoads.push_back(&op);
    }
  }

  Operation *prevBufferLoad = nullptr;
  for (Operation *load : bufferLoads) {
    if (!prevBufferLoad) {
      prevBufferLoad = load;
      continue;
    }

    if (!prevBufferLoad->isBeforeInBlock(load)) {
      prevBufferLoad = load;
      continue;
    }

    // Collect all pure ops between prevBufferLoad and load that the load
    // depends on (transitively).
    llvm::SetVector<Operation *> deps;
    collectDepsInRange(load, prevBufferLoad, deps);

    // Only move if all dependencies are pure (no memory effects).
    if (hasMemoryEffects(deps)) {
      prevBufferLoad = load;
      continue;
    }

    // Move deps in topological order, then the load itself.
    // Since deps are collected via DFS, we need to sort them by their
    // original position in the block to maintain valid SSA ordering.
    SmallVector<Operation *> sortedDeps(deps.begin(), deps.end());
    llvm::sort(sortedDeps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    Operation *insertAfter = prevBufferLoad;
    for (Operation *dep : sortedDeps) {
      dep->moveAfter(insertAfter);
      insertAfter = dep;
    }
    load->moveAfter(insertAfter);

    prevBufferLoad = load;
  }
}

struct ROCDLGroupBufferLoadsPass final
    : impl::ROCDLGroupBufferLoadsPassBase<ROCDLGroupBufferLoadsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    funcOp.walk([](Block *block) { groupBufferLoadsInBlock(*block); });
  }
};

} // namespace

} // namespace mlir::iree_compiler
