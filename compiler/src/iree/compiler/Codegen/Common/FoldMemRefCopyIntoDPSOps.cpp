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

// Finds the storage root of a memref view. For example, both %alloc and
// `memref.subview %alloc` have %alloc as their root.
static Value getMemRefViewRoot(Value value) {
  auto memrefValue = dyn_cast<MemrefValue>(value);
  if (!memrefValue) {
    return value;
  }
  return memref::skipViewLikeOps(memrefValue);
}

static bool isDerivedFromLocalAlloc(Value value) {
  Operation *root = getMemRefViewRoot(value).getDefiningOp();
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root);
}

static std::optional<FlatSymbolRefAttr> getMemRefGlobalRoot(Value value) {
  auto getGlobalOp =
      getMemRefViewRoot(value).getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobalOp) {
    return std::nullopt;
  }
  return getGlobalOp.getNameAttr();
}

// A HAL binding may be hidden by bufferization casts and views. Walk through
// those producers so that, for example, a subview of binding(0) is still
// classified as binding(0).
static std::optional<IREE::HAL::InterfaceBindingSubspanOp>
getInterfaceBindingRoot(Value value) {
  auto typedValue = dyn_cast<TypedValue<MemRefType>>(value);
  if (!typedValue) {
    return std::nullopt;
  }
  return getSourceSubspanMemref(typedValue);
}

// Answers whether two memrefs may address the same storage. MLIR's local alias
// analysis does not model IREE executable bindings and does not distinguish
// all memref globals, so refine those roots before falling back to it:
//
//   lhs root       rhs root       result
//   binding(0)     binding(1)     no alias
//   binding(0)     binding(0)     may alias (subspan offsets may overlap)
//   @global_a      @global_b      no alias
//   binding/global local alloc    no alias
//   unknown        unknown        MLIR AliasAnalysis
//
// A "local alloc" includes its view-like derivatives. Unknown roots remain
// conservative because they may be function arguments or unrecognized aliases.
static bool mayAlias(Value lhs, Value rhs, AliasAnalysis &aliasAnalysis) {
  if (lhs == rhs) {
    return true;
  }

  std::optional<IREE::HAL::InterfaceBindingSubspanOp> lhsSubspan =
      getInterfaceBindingRoot(lhs);
  std::optional<IREE::HAL::InterfaceBindingSubspanOp> rhsSubspan =
      getInterfaceBindingRoot(rhs);
  if (lhsSubspan && rhsSubspan) {
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

// MLIR mod/ref recognizes operations that cannot access the target. Otherwise
// inspect memref operands with mayAlias because generic mod/ref cannot use the
// HAL binding and global distinctions above. An operation with recursive or
// unbound addressable effects remains conservative.
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

// Retargeting a DPS op changes its writes from the private temporary to the
// final target. Reject inputs that may alias that target: an op that reads
// %target while updating %temp would otherwise start reading its own writes.
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
    // Forward the temporary destination produced by bufferization:
    //
    //   copy %source, %temp             copy %source, %target
    //   dps_op outs(%temp)       -->    dps_op outs(%target)
    //   copy %temp, %target
    //
    // The checks below require one init copy, one DPS writer, and one output
    // copy in program order. They also prove that moving the target write to
    // the input-copy position cannot change any intervening memory access.
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
