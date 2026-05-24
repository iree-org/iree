// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOLDMEMREFCOPYINTODPSOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Peel view-like memref producers to the allocation, global, or interface
// binding that defines the addressed storage. This deliberately relies on MLIR's
// ViewLikeOpInterface instead of assuming that every one-memref-operand op is an
// aliasing view.
static Value getMemRefViewRoot(Value value) {
  auto memrefValue = dyn_cast<MemrefValue>(value);
  if (!memrefValue) {
    return value;
  }
  return memref::skipViewLikeOps(memrefValue);
}

// Local allocs are private storage for the dispatch body. They cannot alias HAL
// interface bindings or memref globals, and generic MLIR alias analysis handles
// local-alloc-vs-local-alloc cases.
static bool isDerivedFromLocalAlloc(Value value) {
  Operation *root = getMemRefViewRoot(value).getDefiningOp();
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root);
}

// Return the symbol for a memref global after peeling view-like aliases. Globals
// with different symbols are distinct storage objects, which generic MLIR alias
// analysis does not infer from memref.get_global by itself.
static std::optional<FlatSymbolRefAttr> getMemRefGlobalRoot(Value value) {
  auto getGlobalOp =
      getMemRefViewRoot(value).getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobalOp) {
    return std::nullopt;
  }
  return getGlobalOp.getNameAttr();
}

// Reuse IREE's existing source-subspan walk for memrefs derived from HAL
// interface bindings. Binding identity is the IREE-specific alias fact this pass
// needs on top of generic MLIR local alias analysis.
static std::optional<IREE::HAL::InterfaceBindingSubspanOp>
getSourceSubspan(Value value) {
  auto typedValue = dyn_cast<TypedValue<MemRefType>>(value);
  if (!typedValue) {
    return std::nullopt;
  }
  return getSourceSubspanMemref(typedValue);
}

// Conservative alias predicate for the safety checks in this pass.
//
// - Use explicit IREE facts for HAL interface bindings and memref globals.
// - Defer the remaining cases to MLIR AliasAnalysis.
static bool mayAlias(Value lhs, Value rhs, AliasAnalysis &aliasAnalysis) {
  if (lhs == rhs) {
    return true;
  }

  std::optional<IREE::HAL::InterfaceBindingSubspanOp> lhsSubspan =
      getSourceSubspan(lhs);
  std::optional<IREE::HAL::InterfaceBindingSubspanOp> rhsSubspan =
      getSourceSubspan(rhs);
  if (lhsSubspan && rhsSubspan) {
    // Different interface bindings cannot alias. Treat different subspans of
    // the same binding conservatively because offsets may be dynamic.
    return lhsSubspan->getBinding() == rhsSubspan->getBinding();
  }

  if (lhsSubspan || rhsSubspan) {
    Value other = lhsSubspan ? rhs : lhs;
    return !isDerivedFromLocalAlloc(other) && !getMemRefGlobalRoot(other);
  }

  std::optional<FlatSymbolRefAttr> lhsGlobal = getMemRefGlobalRoot(lhs);
  std::optional<FlatSymbolRefAttr> rhsGlobal = getMemRefGlobalRoot(rhs);
  if (lhsGlobal && rhsGlobal) {
    return *lhsGlobal == *rhsGlobal;
  }
  if (lhsGlobal || rhsGlobal) {
    Value other = lhsGlobal ? rhs : lhs;
    return !isDerivedFromLocalAlloc(other);
  }

  return !aliasAnalysis.alias(lhs, rhs).isNo();
}

// Check whether an intervening operation may access the final target. MLIR
// mod/ref handles effect-free ops and unknown effects; memref operands are then
// refined with the IREE-specific alias facts above.
static bool opMayAccessTarget(Operation *op, Value target,
                              AliasAnalysis &aliasAnalysis) {
  if (aliasAnalysis.getModRef(op, target).isNoModRef()) {
    return false;
  }

  for (Value operand : op->getOperands()) {
    if (isa<MemRefType>(operand.getType()) &&
        mayAlias(operand, target, aliasAnalysis)) {
      return true;
    }
  }

  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface || op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    return true;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectInterface.getEffects(effects);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }
    if (!effect.getValue() && effect.getResource()->isAddressable()) {
      return true;
    }
  }
  return false;
}

// Reject rewrites if any operation between the input and output copies may
// observe or update the final target. The DPS op itself is checked separately,
// so it is allowed in this scan.
static bool hasInterveningTargetAccess(Operation *first, Operation *last,
                                       Operation *allowedOp, Value target,
                                       AliasAnalysis &aliasAnalysis) {
  for (Operation *op = first->getNextNode(); op && op != last;
       op = op->getNextNode()) {
    if (op == allowedOp) {
      continue;
    }
    if (opMayAccessTarget(op, target, aliasAnalysis)) {
      return true;
    }
  }
  return false;
}

// The final target is about to replace the temporary DPS init, so none of the
// values read by the DPS op can alias it. The original copy source is included
// because preserving a partial update requires copying it into the target first.
static bool targetMayAliasDpsReads(DestinationStyleOpInterface dpsOp,
                                   OpOperand *forwardedInitOperand,
                                   Value copySource, Value target,
                                   AliasAnalysis &aliasAnalysis) {
  if (mayAlias(copySource, target, aliasAnalysis)) {
    return true;
  }

  for (OpOperand &operand : dpsOp->getOpOperands()) {
    if (&operand == forwardedInitOperand) {
      continue;
    }
    if (isa<MemRefType>(operand.get().getType()) &&
        mayAlias(operand.get(), target, aliasAnalysis)) {
      return true;
    }
  }
  return false;
}

struct FoldTemporaryCopyIntoDpsOp final : OpRewritePattern<memref::CopyOp> {
  FoldTemporaryCopyIntoDpsOp(MLIRContext *context, AliasAnalysis &aliasAnalysis)
      : OpRewritePattern(context), aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(memref::CopyOp copyOut,
                                PatternRewriter &rewriter) const override {
    auto allocOp = copyOut.getSource().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    memref::CopyOp copyIn;
    DestinationStyleOpInterface dpsOp;
    OpOperand *forwardedInitOperand = nullptr;
    for (Operation *user : allocOp->getUsers()) {
      if (user == copyOut.getOperation()) {
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == allocOp.getResult() && !copyIn) {
          copyIn = copy;
          continue;
        }
        return failure();
      }
      if (auto candidate = dyn_cast<DestinationStyleOpInterface>(user)) {
        if (candidate.getNumDpsInits() != 1 || dpsOp) {
          return failure();
        }
        OpOperand *initOperand = candidate.getDpsInitOperand(0);
        if (initOperand->get() != allocOp.getResult()) {
          return failure();
        }
        dpsOp = candidate;
        forwardedInitOperand = initOperand;
        continue;
      }
      return failure();
    }

    if (!copyIn || !dpsOp || !forwardedInitOperand) {
      return failure();
    }
    if (copyIn->getBlock() != dpsOp->getBlock() ||
        dpsOp->getBlock() != copyOut->getBlock()) {
      return failure();
    }
    if (!copyIn->isBeforeInBlock(dpsOp) || !dpsOp->isBeforeInBlock(copyOut)) {
      return failure();
    }

    Value finalTarget = copyOut.getTarget();
    if (targetMayAliasDpsReads(dpsOp, forwardedInitOperand, copyIn.getSource(),
                               finalTarget, aliasAnalysis)) {
      return failure();
    }
    if (hasInterveningTargetAccess(copyIn, copyOut, dpsOp, finalTarget,
                                   aliasAnalysis)) {
      return failure();
    }

    rewriter.setInsertionPoint(copyIn);
    memref::CopyOp::create(rewriter, copyIn.getLoc(), copyIn.getSource(),
                           finalTarget);
    forwardedInitOperand->set(finalTarget);

    rewriter.eraseOp(copyOut);
    rewriter.eraseOp(copyIn);
    if (allocOp->use_empty()) {
      rewriter.eraseOp(allocOp);
    }
    return success();
  }

private:
  AliasAnalysis &aliasAnalysis;
};

struct FoldMemRefCopyIntoDPSOpsPass final
    : impl::FoldMemRefCopyIntoDPSOpsPassBase<FoldMemRefCopyIntoDPSOpsPass> {
  using FoldMemRefCopyIntoDPSOpsPassBase::FoldMemRefCopyIntoDPSOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    patterns.add<FoldTemporaryCopyIntoDpsOp>(&getContext(), aliasAnalysis);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
