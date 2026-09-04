// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOLDMEMREFCOPYINTODPSOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Peel view-like memref producers to the allocation, global, or interface
// binding that defines the addressed storage. This deliberately relies on
// MLIR's ViewLikeOpInterface instead of assuming that every one-memref-operand
// op is an aliasing view.
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

// Return the symbol for a memref global after peeling view-like aliases.
// Globals with different symbols are distinct storage objects, which generic
// MLIR alias analysis does not infer from memref.get_global by itself.
static std::optional<FlatSymbolRefAttr> getMemRefGlobalRoot(Value value) {
  auto getGlobalOp =
      getMemRefViewRoot(value).getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobalOp) {
    return std::nullopt;
  }
  return getGlobalOp.getNameAttr();
}

// Reuse IREE's existing source-subspan walk for memrefs derived from HAL
// interface bindings. Binding identity is the IREE-specific alias fact this
// pass needs on top of generic MLIR local alias analysis.
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
// because preserving a partial update requires copying it into the target
// first.
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

// Return the static element count only when every dimension is known and the
// product fits in int64_t. Unknown or overflowing shapes make the overwrite
// proof fail conservatively.
static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim) || dim < 0) {
      return std::nullopt;
    }
    if (dim == 0) {
      return 0;
    }
    if (numElements > std::numeric_limits<int64_t>::max() / dim) {
      return std::nullopt;
    }
    numElements *= dim;
  }
  return numElements;
}

// A linalg op fully overwrites its init when the output element is not read and
// every output point is written exactly once. We prove that only for fills,
// copies, and parallel identity-mapped generics.
static bool linalgOpFullyOverwritesInit(linalg::LinalgOp linalgOp,
                                        OpOperand *initOperand) {
  Operation *op = linalgOp.getOperation();
  if (isa<linalg::FillOp, linalg::CopyOp>(op)) {
    return true;
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                    linalg::isParallelIterator)) {
    return false;
  }

  AffineMap initMap = genericOp.getMatchingIndexingMap(initOperand);
  auto initType = dyn_cast<MemRefType>(initOperand->get().getType());
  if (!initType || initMap.getNumDims() != initType.getRank() ||
      initMap.getNumResults() != initType.getRank() || !initMap.isIdentity()) {
    return false;
  }

  Value blockArg = genericOp.getMatchingBlockArgument(initOperand);
  return blockArg && blockArg.use_empty();
}

// Extract constant integer data from a memref global. Scatter overwrite proofs
// require concrete indices so that we can prove every destination element is
// covered exactly once.
static DenseIntElementsAttr getDenseGlobalIntegerElements(Value value) {
  auto getGlobal = value.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    return {};
  }

  auto globalOp = dyn_cast_if_present<memref::GlobalOp>(
      SymbolTable::lookupNearestSymbolFrom(getGlobal, getGlobal.getNameAttr()));
  if (!globalOp || !globalOp.getConstant()) {
    return {};
  }
  return dyn_cast_if_present<DenseIntElementsAttr>(
      globalOp.getInitialValueAttr());
}

// Prove that a scatter writes the entire init without reading old values. This
// is intentionally narrow: unique scalar updates, identity dimension map,
// static shapes, constant indices, and an unused old-value block argument.
static bool scatterFullyOverwritesInit(IREE::LinalgExt::ScatterOp scatterOp,
                                       OpOperand *initOperand) {
  if (initOperand != scatterOp.getDpsInitOperand(0)) {
    return false;
  }
  if (scatterOp.getMask() || !scatterOp.getUniqueIndices()) {
    return false;
  }

  Block &body = scatterOp.getRegion().front();
  if (body.getNumArguments() != 2 || !body.getArgument(1).use_empty()) {
    return false;
  }

  ShapedType originalType = scatterOp.getOriginalType();
  ShapedType updateType = scatterOp.getUpdateType();
  ShapedType indicesType = scatterOp.getIndicesType();
  if (!originalType.hasStaticShape() || !updateType.hasStaticShape() ||
      !indicesType.hasStaticShape()) {
    return false;
  }

  int64_t indexDepth = scatterOp.getIndexDepth();
  int64_t batchRank = scatterOp.getBatchRank();
  if (!scatterOp.isScalarUpdate()) {
    return false;
  }
  for (auto [index, dimension] : llvm::enumerate(scatterOp.getDimensionMap())) {
    if (dimension != static_cast<int64_t>(index)) {
      return false;
    }
  }
  bool omittedIndexDepthDim = indicesType.getRank() == batchRank;
  if (omittedIndexDepthDim && indexDepth != 1) {
    return false;
  }

  std::optional<int64_t> batchElementCount =
      getStaticNumElements(updateType.getShape().take_front(batchRank));
  std::optional<int64_t> indexedElementCount =
      getStaticNumElements(originalType.getShape().take_front(indexDepth));
  if (!batchElementCount || !indexedElementCount ||
      *batchElementCount != *indexedElementCount) {
    return false;
  }

  DenseIntElementsAttr indicesAttr =
      getDenseGlobalIntegerElements(scatterOp.getIndices());
  if (!indicesAttr) {
    return false;
  }

  int64_t elementsPerIndex = omittedIndexDepthDim ? 1 : indexDepth;
  if (*batchElementCount >
      std::numeric_limits<int64_t>::max() / elementsPerIndex) {
    return false;
  }
  int64_t expectedIndexElementCount = *batchElementCount * elementsPerIndex;
  if (indicesAttr.getNumElements() != expectedIndexElementCount) {
    return false;
  }

  SmallVector<int64_t> indexValues;
  indexValues.reserve(expectedIndexElementCount);
  for (APInt value : indicesAttr.getValues<APInt>()) {
    if (value.getBitWidth() > 64) {
      return false;
    }
    indexValues.push_back(value.getSExtValue());
  }

  ArrayRef<int64_t> dimensionMap = scatterOp.getDimensionMap();
  ArrayRef<int64_t> originalShape = originalType.getShape();
  llvm::DenseSet<int64_t> covered;
  covered.reserve(*batchElementCount);
  for (int64_t batchIndex = 0; batchIndex < *batchElementCount; ++batchIndex) {
    SmallVector<int64_t> destinationIndices(indexDepth, 0);
    if (omittedIndexDepthDim) {
      destinationIndices[dimensionMap[0]] = indexValues[batchIndex];
    } else {
      for (int64_t index = 0; index < indexDepth; ++index) {
        destinationIndices[dimensionMap[index]] =
            indexValues[batchIndex * indexDepth + index];
      }
    }

    int64_t linearIndex = 0;
    for (int64_t dim = 0; dim < indexDepth; ++dim) {
      int64_t coordinate = destinationIndices[dim];
      int64_t dimSize = originalShape[dim];
      if (coordinate < 0 || coordinate >= dimSize ||
          linearIndex > std::numeric_limits<int64_t>::max() / dimSize) {
        return false;
      }
      linearIndex = linearIndex * dimSize + coordinate;
    }
    if (!covered.insert(linearIndex).second) {
      return false;
    }
  }

  return covered.size() == static_cast<size_t>(*indexedElementCount);
}

// Decide whether the initial copy into the temporary can be dropped entirely.
// If this returns false, the pass still forwards the temporary into the final
// target but preserves the source copy before the DPS op.
static bool dpsInitCopyCanBeElided(DestinationStyleOpInterface dpsOp,
                                   OpOperand *forwardedInitOperand) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dpsOp.getOperation())) {
    return linalgOpFullyOverwritesInit(linalgOp, forwardedInitOperand);
  }
  if (auto scatterOp =
          dyn_cast<IREE::LinalgExt::ScatterOp>(dpsOp.getOperation())) {
    return scatterFullyOverwritesInit(scatterOp, forwardedInitOperand);
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
    if (!dpsInitCopyCanBeElided(dpsOp, forwardedInitOperand)) {
      memref::CopyOp::create(rewriter, copyIn.getLoc(), copyIn.getSource(),
                             finalTarget);
    }
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
