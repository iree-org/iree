// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REINSERTSWIZZLEHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ReinsertSwizzleHintsPass final
    : impl::ReinsertSwizzleHintsPassBase<ReinsertSwizzleHintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Traces a memref value backward through view-like ops, memref.cast, and
/// scf.for iter_args/results to find the root memref.alloc.
static memref::AllocOp traceToAllocation(Value val) {
  DenseSet<Value> visited;
  SmallVector<Value> worklist = {val};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      auto blockArg = cast<BlockArgument>(current);
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      unsigned argIdx = blockArg.getArgNumber();
      if (!forOp || argIdx == 0) {
        continue;
      }
      worklist.push_back(forOp.getInitArgs()[argIdx - 1]);
    } else if (auto allocOp = dyn_cast<memref::AllocOp>(defOp)) {
      return allocOp;
    } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned resultIdx = cast<OpResult>(current).getResultNumber();
      worklist.push_back(forOp.getInitArgs()[resultIdx]);
    } else if (isa<ViewLikeOpInterface>(defOp)) {
      worklist.push_back(defOp->getOperand(0));
    } else if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      worklist.push_back(castOp.getSource());
    }
  }
  return nullptr;
}

/// Returns the swizzle attribute on the alloc that val traces to, using
/// cache to avoid repeated tracing.
static IREE::Codegen::SwizzleAttrInterface lookupSwizzleAttr(
    Value val,
    DenseMap<Operation *, IREE::Codegen::SwizzleAttrInterface> &cache) {
  memref::AllocOp allocOp = traceToAllocation(val);
  if (!allocOp) {
    return nullptr;
  }
  auto it = cache.find(allocOp.getOperation());
  if (it != cache.end()) {
    return it->second;
  }
  auto swizzle = allocOp->getAttrOfType<IREE::Codegen::SwizzleAttrInterface>(
      "iree_codegen.swizzle");
  cache[allocOp.getOperation()] = swizzle;
  return swizzle;
}

/// Wraps |source| with collapse_shape -> swizzle_hint -> expand_shape so that
/// downstream ResolveSwizzleHints can apply the XOR transform. For 1D memrefs,
/// only the swizzle_hint is inserted.
static Value insertSwizzleHint(IRRewriter &rewriter, Location loc, Value source,
                               IREE::Codegen::SwizzleAttrInterface swizzle) {
  auto sourceType = cast<MemRefType>(source.getType());

  Value hintInput = source;
  SmallVector<ReassociationIndices> reassoc;

  if (sourceType.getRank() > 1) {
    reassoc.push_back(
        llvm::to_vector(llvm::seq<int64_t>(0, sourceType.getRank())));
    hintInput = memref::CollapseShapeOp::create(rewriter, loc, source, reassoc);
  }

  auto hintOp =
      IREE::Codegen::SwizzleHintOp::create(rewriter, loc, hintInput, swizzle);

  if (sourceType.getRank() > 1) {
    return memref::ExpandShapeOp::create(rewriter, loc, sourceType.getShape(),
                                         hintOp.getResult(), reassoc);
  }
  return hintOp.getResult();
}

void ReinsertSwizzleHintsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());
  DenseMap<Operation *, IREE::Codegen::SwizzleAttrInterface> swizzleCache;

  // For each vector.load/store whose base traces to a swizzled alloc, wrap the
  // base with collapse_shape -> swizzle_hint -> expand_shape.
  funcOp.walk([&](Operation *op) {
    Value base;
    if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      base = loadOp.getBase();
    } else if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
      base = storeOp.getBase();
    } else {
      return;
    }
    IREE::Codegen::SwizzleAttrInterface swizzle =
        lookupSwizzleAttr(base, swizzleCache);
    if (!swizzle) {
      return;
    }
    rewriter.setInsertionPoint(op);
    Value wrapped = insertSwizzleHint(rewriter, op->getLoc(), base, swizzle);
    if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      loadOp.getBaseMutable().assign(wrapped);
    } else if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
      storeOp.getBaseMutable().assign(wrapped);
    }
  });

  // Clean up the swizzle attributes from allocs now that hints are inserted.
  funcOp.walk([](memref::AllocOp allocOp) {
    allocOp->removeAttr("iree_codegen.swizzle");
  });
}

} // namespace mlir::iree_compiler
