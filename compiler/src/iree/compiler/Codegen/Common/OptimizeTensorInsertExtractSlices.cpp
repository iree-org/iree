// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-codegen-optimize-tensor-insert-extract-slices"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_OPTIMIZETENSORINSERTEXTRACTSLICESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kTransferReadTag = "loop_carried_transfer_read";
static constexpr StringLiteral kTransferWriteTag =
    "loop_carried_transfer_write";

class OptimizeTensorInsertExtractSlicesPass final
    : public impl::OptimizeTensorInsertExtractSlicesPassBase<
          OptimizeTensorInsertExtractSlicesPass> {
  using impl::OptimizeTensorInsertExtractSlicesPassBase<
      OptimizeTensorInsertExtractSlicesPass>::
      OptimizeTensorInsertExtractSlicesPassBase;

public:
  void runOnOperation() override;
};

// Check if this insertion is loop invariant except it's source.
// We would also be okay as long as the destination is loop invariant,
// but we would have to do some cloning, so we don't do it here.
static bool canBeHoisted(LoopLikeOpInterface loopLike,
                         SubsetInsertionOpInterface insertion) {
  // Do not move terminators.
  if (insertion->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (insertion->isAncestor(
              operand.get().getParentRegion()->getParentOp())) {
        continue;
      }
      if (!loopLike.isDefinedOutsideOfLoop(operand.get()) &&
          &operand != &insertion.getSourceOperand()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  };
  return !insertion->walk(walkFn).wasInterrupted();
}

/// Return the newly created loop op (that has extra iter_args) or the original
/// loop op if nothing was hoisted.
static LoopLikeOpInterface
hoistLoopInvariantSubsetAtIterArg(RewriterBase &rewriter,
                                  LoopLikeOpInterface loopLike, int64_t idx) {
  // Get subset insertion of this yielded arg.
  auto insertion = loopLike.getYieldedValues()[idx]
                       .getDefiningOp<SubsetInsertionOpInterface>();
  if (!insertion) {
    return loopLike;
  }
  if (!canBeHoisted(loopLike, insertion)) {
    return loopLike;
  }

  bool changed = true;
  // The below `while` loop is a WAR for the core issue that is unidentified.
  // To avoid infinite loops, limit number of iterations to 10.
  int numIterations = 0;
  while (changed && numIterations < 10) {
    numIterations++;
    changed = false;
    // Get all subset extraction uses of this iter_arg and try to hoist them
    // out of the loop.
    for (Operation *op : loopLike.getRegionIterArgs()[idx].getUsers()) {
      auto extraction = dyn_cast<SubsetExtractionOpInterface>(op);
      if (!extraction) {
        continue;
      }

      // Check if this extraction is operating on the same subset as the
      // insertion.
      bool equivalent = extraction.operatesOnEquivalentSubset(
          insertion, [](Value v1, Value v2) {
            // The callback to this method checks if the given two values are
            // aliasing tensors/buffers from which the subset slice comes from.
            // For our case, we only care if the slices are same, so we can
            // always return true.
            return true;
          });

      if (!equivalent) {
        continue;
      }

      // Hoist out the extraction/insertion ops.
      NewYieldValuesFn newYieldValuesFn =
          [&](OpBuilder &b, Location loc,
              ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
        return {insertion.getSourceOperand().get()};
      };

      // replaceInitOperandUsesInLoop is set to true S.T we will use new IV
      // instead of hoisted out extract.
      FailureOr<LoopLikeOpInterface> newLoop =
          loopLike.replaceWithAdditionalYields(
              rewriter, extraction.getResult(),
              /*replaceInitOperandUsesInLoop=*/true, newYieldValuesFn);
      if (failed(newLoop)) {
        return loopLike;
      }
      loopLike = *newLoop;

      BlockArgument iterArg = loopLike.getRegionIterArgs()[idx];
      OpResult loopResult = loopLike.getTiedLoopResult(iterArg);
      OpResult newLoopResult = loopLike.getLoopResults()->back();
      rewriter.moveOpBefore(extraction, loopLike);

      // Hoist the extraction/insertion ops.
      extraction.getSourceOperand().set(
          loopLike.getTiedLoopInit(iterArg)->get());

      // Clone the insertion to outside the not removing the final insertion, as
      // it still can be used by other extraction ops loop.
      rewriter.setInsertionPointAfter(loopLike);
      SubsetInsertionOpInterface newInsertion =
          cast<SubsetInsertionOpInterface>(
              rewriter.clone(*insertion.getOperation()));

      rewriter.replaceAllUsesWith(loopResult,
                                  newInsertion.getUpdatedDestination());
      newInsertion.getSourceOperand().set(newLoopResult);

      // loopLike changed, restart the check.
      changed = true;
      break;
    }
  }

  return loopLike;
}

/// The task of loop invariant subset hoisting as transformation is to find
/// a subset being used by a loop, which is "loop invariant", i.e. the loop
/// always works on that subset, instead of the whole set. Example:
///
/// for %i = 0 to 128 iter_args(%t = %init) {
///   %a = extract_slice %t[0, 0][8, 8]
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %a, %b
///   %out = %insert_slice %t[0, 0][8, 8]
///   yield %out
/// }
///
/// In this example, the loop is only operating on a loop invariant subset
/// of %t, which allows us to hoist out the extract_slice/insert_slice out
/// of the loop, and pass the subset as an iter_arg.
///
/// %a = extract_slice %init[0, 0][8, 8]
/// %loop = for %i = 0 to 128 iter_args(%t = %a) {
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %t, %b
///   yield %c
/// }
/// %out = insert_slice %loop into %init[0, 0][8, 8]
///
/// This hoisting only works when we are working on the same subset of the same
/// tensor, because the complement of the subset could have been updated,
/// but we don't know about it, so we need to preserve it.
///
/// However, if the destination of the insertion is a loop invariant tensor,
/// we do not need to preserve the complement of the subset, so we can still do
/// the hoisting. Example:
///
/// for %i = 0 to 128 iter_args(%t = %init) {
///   %a = extract_slice %t[0, 0][8, 8]
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %a, %b
///   %out = %insert_slice %init2[0, 0][8, 8]
///   yield %out
/// }
///
/// %a = extract_slice %init[0, 0][8, 8]
/// %loop = for %i = 0 to 128 iter_args(%t = %a) {
///   %b = extract_slice %t2[0, %i][8, 8]
///   %c = add %t, %b
///   yield %c
/// }
/// %out = insert_slice %loop into %init2[0, 0][8, 8]
///
/// The function implements the later transformation.
///
/// TODO (Groverkss): Improve upstream subset hoisting to account for this. I
/// think there is a more general way to handle this.
void hoistSubsetWithLoopInvariantTensor(RewriterBase &rewriter,
                                        LoopLikeOpInterface loopLike) {
  for (int64_t i = 0;
       i < static_cast<int64_t>(loopLike.getRegionIterArgs().size()); ++i) {
    loopLike = hoistLoopInvariantSubsetAtIterArg(rewriter, loopLike, i);
  }
}

static bool
hasOnlyLoopInvariantOperandsExcept(LoopLikeOpInterface loopLike, Operation *op,
                                   ArrayRef<OpOperand *> allowedOperands) {
  return llvm::all_of(op->getOpOperands(), [&](OpOperand &operand) {
    if (llvm::is_contained(allowedOperands, &operand)) {
      return true;
    }
    return loopLike.isDefinedOutsideOfLoop(operand.get());
  });
}

static bool areEquivalentMasks(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs || lhs.getType() != rhs.getType()) {
    return false;
  }
  auto lhsCreateMask = lhs.getDefiningOp<vector::CreateMaskOp>();
  auto rhsCreateMask = rhs.getDefiningOp<vector::CreateMaskOp>();
  if (lhsCreateMask && rhsCreateMask) {
    return lhsCreateMask.getOperands() == rhsCreateMask.getOperands();
  }
  auto lhsConstantMask = lhs.getDefiningOp<vector::ConstantMaskOp>();
  auto rhsConstantMask = rhs.getDefiningOp<vector::ConstantMaskOp>();
  return lhsConstantMask && rhsConstantMask &&
         lhsConstantMask.getMaskDimSizes() == rhsConstantMask.getMaskDimSizes();
}

static bool
areEquivalentTransferAccesses(Type lhsVectorType, ValueRange lhsIndices,
                              AffineMap lhsPermutationMap, Value lhsMask,
                              Type rhsVectorType, ValueRange rhsIndices,
                              AffineMap rhsPermutationMap, Value rhsMask) {
  return lhsVectorType == rhsVectorType && lhsIndices == rhsIndices &&
         lhsPermutationMap == rhsPermutationMap &&
         areEquivalentMasks(lhsMask, rhsMask);
}

static bool areEquivalentTransferReadWrite(vector::TransferReadOp readOp,
                                           vector::TransferWriteOp writeOp) {
  if (readOp.isMasked() || writeOp.isMasked()) {
    return false;
  }
  return areEquivalentTransferAccesses(
      readOp.getType(), readOp.getIndices(), readOp.getPermutationMap(),
      readOp.getMask(), writeOp.getVector().getType(), writeOp.getIndices(),
      writeOp.getPermutationMap(), writeOp.getMask());
}

static bool areEquivalentTransferWrites(vector::TransferWriteOp lhs,
                                        vector::TransferWriteOp rhs) {
  if (lhs.isMasked() || rhs.isMasked()) {
    return false;
  }
  return areEquivalentTransferAccesses(
      lhs.getVector().getType(), lhs.getIndices(), lhs.getPermutationMap(),
      lhs.getMask(), rhs.getVector().getType(), rhs.getIndices(),
      rhs.getPermutationMap(), rhs.getMask());
}

static Value getLoopCarriedTransferValue(OpBuilder &builder, Location loc,
                                         TypedValue<VectorType> mask, Value pad,
                                         Value valueToStore) {
  if (!mask) {
    return valueToStore;
  }
  Value padVector = vector::BroadcastOp::create(
      builder, pad.getLoc(), cast<VectorType>(valueToStore.getType()), pad);
  return arith::SelectOp::create(builder, loc, mask, valueToStore, padVector);
}

/// Promote a tensor iter_arg to a vector iter_arg when the tensor is only
/// accessed through equivalent transfer_read/transfer_write ops.
///
///   %loop = scf.for ... iter_args(%t = %init) -> tensor<...> {
///     %old = vector.transfer_read %t[%c0], %pad, %mask
///     %new = arith.addf %old, %step
///     %next = vector.transfer_write %new, %t[%c0], %mask
///     scf.yield %next
///   }
///
/// becomes:
///
///   %init_vec = vector.transfer_read %init[%c0], %pad, %mask
///   %loop = scf.for ... iter_args(%v = %init_vec) -> vector<...> {
///     %new = arith.addf %v, %step
///     %yield = arith.select %mask, %new, splat(%pad)
///     scf.yield %yield
///   }
///
/// The select is the important masked-transfer semantics. For masked-off
/// lanes, transfer_read returns padding, while transfer_write preserves the
/// tensor. A generic subset hoist would carry `%new` directly, which is only
/// equivalent for unmasked transfers.
struct MarkHoistableLoopCarriedTransferIterArg final
    : OpRewritePattern<vector::TransferWriteOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    std::optional<HoistableTransferIterArg> candidate =
        getHoistableTransferIterArg(writeOp);
    if (!candidate) {
      return failure();
    }
    return markLoopCarriedTransferIterArgHoistable(rewriter, *candidate);
  }

private:
  struct HoistableTransferIterArg {
    BlockArgument iterArg;
    Value initArg;
    vector::TransferWriteOp write;
    SmallVector<vector::TransferReadOp> reads;
  };

  std::optional<HoistableTransferIterArg>
  getHoistableTransferIterArg(vector::TransferWriteOp writeOp) const {
    auto loopLike =
        dyn_cast_if_present<LoopLikeOpInterface>(writeOp->getParentOp());
    if (!loopLike || !writeOp.hasPureTensorSemantics() ||
        writeOp.hasOutOfBoundsDim() || !writeOp->hasOneUse()) {
      return std::nullopt;
    }

    auto yieldedMutable = loopLike.getYieldedValuesMutable();
    if (!yieldedMutable) {
      return std::nullopt;
    }

    std::optional<unsigned> index;
    for (auto [i, yieldOperand] : llvm::enumerate(*yieldedMutable)) {
      if (yieldOperand.get() == writeOp.getResult()) {
        index = i;
        break;
      }
    }
    if (!index) {
      return std::nullopt;
    }

    BlockArgument iterArg = loopLike.getRegionIterArgs()[*index];
    OpOperand *initArg = loopLike.getTiedLoopInit(iterArg);
    if (!initArg || writeOp.getBase() != iterArg) {
      return std::nullopt;
    }
    // Avoid marking inner loops whose init is carried by an enclosing loop:
    // the conversion body would capture that outer iter_arg and can become
    // self-referential when the conversion is hoisted through the outer loop.
    if (auto initBlockArg = dyn_cast<BlockArgument>(initArg->get())) {
      if (dyn_cast_if_present<LoopLikeOpInterface>(
              initBlockArg.getOwner()->getParentOp())) {
        return std::nullopt;
      }
    }

    if (!hasOnlyLoopInvariantOperandsExcept(
            loopLike, writeOp,
            {&writeOp.getValueToStoreMutable(), &writeOp.getBaseMutable()})) {
      return std::nullopt;
    }

    SmallVector<vector::TransferReadOp> reads;
    for (Operation *user : iterArg.getUsers()) {
      if (user == writeOp.getOperation()) {
        continue;
      }
      auto readOp = dyn_cast<vector::TransferReadOp>(user);
      if (!readOp || !readOp.hasPureTensorSemantics() ||
          readOp.getBase() != iterArg || readOp.hasOutOfBoundsDim()) {
        return std::nullopt;
      }
      if (!areEquivalentTransferReadWrite(readOp, writeOp)) {
        return std::nullopt;
      }
      if (!reads.empty() && readOp.getPadding() != reads.front().getPadding()) {
        return std::nullopt;
      }
      if (!hasOnlyLoopInvariantOperandsExcept(loopLike, readOp,
                                              {&readOp.getBaseMutable()})) {
        return std::nullopt;
      }
      reads.push_back(readOp);
    }

    if (reads.empty()) {
      return std::nullopt;
    }
    return HoistableTransferIterArg{iterArg, initArg->get(), writeOp, reads};
  }

  LogicalResult markLoopCarriedTransferIterArgHoistable(
      PatternRewriter &rewriter, HoistableTransferIterArg candidate) const {
    OpBuilder::InsertionGuard guard(rewriter);
    vector::TransferReadOp readOp = candidate.reads.front();
    Operation *insertionPoint = readOp.getOperation();
    for (vector::TransferReadOp candidateRead : candidate.reads) {
      if (candidateRead->isBeforeInBlock(insertionPoint)) {
        insertionPoint = candidateRead.getOperation();
      }
    }

    rewriter.setInsertionPoint(insertionPoint);
    auto readConversion = IREE::Util::HoistableConversionOp::create(
        rewriter, readOp.getLoc(), kTransferReadTag, kTransferWriteTag,
        TypeRange{readOp.getVectorType()}, ValueRange{candidate.iterArg},
        [&](OpBuilder &builder, Location, ValueRange args) {
          auto clonedRead = cast<vector::TransferReadOp>(
              builder.clone(*readOp.getOperation()));
          clonedRead.getBaseMutable().assign(args[0]);
          return SmallVector<Value>{clonedRead.getResult()};
        });

    TypedValue<VectorType> mask = readOp.getMask();
    Value pad = readOp.getPadding();
    for (vector::TransferReadOp read : candidate.reads) {
      rewriter.replaceOp(read, readConversion.getResult(0));
    }

    vector::TransferWriteOp writeOp = candidate.write;
    rewriter.setInsertionPoint(writeOp);
    Value carriedValue = getLoopCarriedTransferValue(
        rewriter, writeOp.getLoc(), mask, pad, writeOp.getVector());
    auto writeConversion = IREE::Util::HoistableConversionOp::create(
        rewriter, writeOp.getLoc(), kTransferWriteTag, kTransferReadTag,
        TypeRange{writeOp.getResult().getType()}, ValueRange{carriedValue},
        [&](OpBuilder &builder, Location, ValueRange args) {
          auto clonedWrite = cast<vector::TransferWriteOp>(
              builder.clone(*writeOp.getOperation()));
          clonedWrite.getValueToStoreMutable().assign(args[0]);
          clonedWrite.getBaseMutable().assign(candidate.initArg);
          return SmallVector<Value>{clonedWrite.getResult()};
        });

    rewriter.replaceOp(writeOp, writeConversion.getResult(0));

    return success();
  }
};

/// Fold tensor.dim of a loop-carried tensor iter_arg back to the loop init
/// tensor when the yielded value is only a chain of shape-preserving
/// vector.transfer_write ops.
///
/// This normalizes masks built from loop-carried tensor dims before transfer
/// folding runs. Without this, equivalent masks may remain structurally
/// different because one is based on the init tensor and another is based on
/// the iter_arg.
///
/// ```mlir
/// %r = scf.for ... iter_args(%acc = %init) -> tensor<?xf32> {
///   %d = tensor.dim %acc, %c0 : tensor<?xf32>
///   %m = vector.create_mask %d : vector<4xi1>
///   %next = vector.transfer_write %v, %acc[%c0], %m
///   scf.yield %next : tensor<?xf32>
/// }
/// ```
///
/// becomes:
///
/// ```mlir
/// %r = scf.for ... iter_args(%acc = %init) -> tensor<?xf32> {
///   %d = tensor.dim %init, %c0 : tensor<?xf32>
///   %m = vector.create_mask %d : vector<4xi1>
///   %next = vector.transfer_write %v, %acc[%c0], %m
///   scf.yield %next : tensor<?xf32>
/// }
/// ```
struct FoldDimOfShapePreservingTransferIterArg final
    : OpRewritePattern<tensor::DimOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto iterArg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!iterArg) {
      return failure();
    }
    auto forOp = dyn_cast<scf::ForOp>(iterArg.getOwner()->getParentOp());
    if (!forOp) {
      return failure();
    }
    OpOperand *initArg = forOp.getTiedLoopInit(iterArg);
    if (!initArg || !isShapePreservingTransferWriteChain(forOp, iterArg)) {
      return failure();
    }

    rewriter.modifyOpInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(initArg->get()); });
    return success();
  }

private:
  bool isShapePreservingTransferWriteChain(scf::ForOp forOp,
                                           BlockArgument iterArg) const {
    OpOperand *yieldedValue = forOp.getTiedLoopYieldedValue(iterArg);
    if (!yieldedValue) {
      return false;
    }
    Value value = yieldedValue->get();
    while (true) {
      if (value == iterArg) {
        return true;
      }
      auto writeOp = value.getDefiningOp<vector::TransferWriteOp>();
      if (!writeOp || !writeOp.hasPureTensorSemantics()) {
        return false;
      }
      value = writeOp.getBase();
    }
  }
};

/// Fold transfer_write chains where a later write overwrites the same tensor
/// slice as an earlier write. For equal masks, masked-off lanes are preserved
/// from the original base by both writes, so the earlier write is not needed
/// for the later result.
///
/// This handles structurally equivalent masks that upstream WAW folding does
/// not recognize after CSE/canonicalization fail to make the mask SSA value the
/// same.
///
/// ```mlir
/// %m0 = vector.create_mask %d : vector<4xi1>
/// %w0 = vector.transfer_write %v0, %t[%c0], %m0
/// %m1 = vector.create_mask %d : vector<4xi1>
/// %w1 = vector.transfer_write %v1, %w0[%c0], %m1
/// ```
///
/// becomes:
///
/// ```mlir
/// %m1 = vector.create_mask %d : vector<4xi1>
/// %w1 = vector.transfer_write %v1, %t[%c0], %m1
/// ```
struct FoldTransferWriteChain final
    : OpRewritePattern<vector::TransferWriteOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    if (!writeOp.hasPureTensorSemantics() || writeOp.hasOutOfBoundsDim()) {
      return failure();
    }
    auto previousWriteOp =
        writeOp.getBase().getDefiningOp<vector::TransferWriteOp>();
    if (!previousWriteOp || !previousWriteOp.hasPureTensorSemantics() ||
        previousWriteOp.hasOutOfBoundsDim() ||
        !areEquivalentTransferWrites(writeOp, previousWriteOp)) {
      return failure();
    }

    rewriter.modifyOpInPlace(writeOp, [&]() {
      writeOp.getBaseMutable().assign(previousWriteOp.getBase());
    });
    if (previousWriteOp->use_empty()) {
      rewriter.eraseOp(previousWriteOp);
    }
    return success();
  }
};

namespace {
struct CastLikeExtractSliceOpFolder final
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeExtractSliceOp(sliceOp) ||
        sliceOp.getSourceType() != sliceOp.getResultType()) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getSource());
    return success();
  }
};

struct CastLikeInsertSliceOpFolder final
    : OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!tensor::isCastLikeInsertSliceOp(sliceOp) ||
        sliceOp.getSourceType() != sliceOp.getResultType()) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getSource());
    return success();
  }
};

/// Retarget vector.transfer_read through an identity tensor slice.
///
/// This is narrower than the general identity-slice folder: it only updates the
/// transfer_read base and only for zero-offset, unit-stride slices with
/// identical source/result types. The purpose is to unblock transfer folding in
/// cleanup phases that intentionally do not run the general
/// fold-identity-slices patterns.
///
/// ```mlir
/// %slice = tensor.extract_slice %t[0] [%d] [1]
///     : tensor<?xf32> to tensor<?xf32>
/// %r = vector.transfer_read %slice[%c0], %pad
///     : tensor<?xf32>, vector<4xf32>
/// ```
///
/// becomes:
///
/// ```mlir
/// %r = vector.transfer_read %t[%c0], %pad
///     : tensor<?xf32>, vector<4xf32>
/// ```
///
/// The same rewrite applies to identity tensor.insert_slice results, replacing
/// the read base with the inserted source tensor.
struct FoldTransferReadOfIdentitySlice final
    : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (!readOp.hasPureTensorSemantics()) {
      return failure();
    }
    Value source = getIdentitySliceSource(readOp.getBase());
    if (!source) {
      return failure();
    }
    rewriter.modifyOpInPlace(readOp,
                             [&]() { readOp.getBaseMutable().assign(source); });
    return success();
  }

private:
  bool isIdentityExtractSliceOp(tensor::ExtractSliceOp extractSliceOp) const {
    return tensor::isCastLikeExtractSliceOp(extractSliceOp) &&
           extractSliceOp.getSourceType() == extractSliceOp.getResultType() &&
           extractSliceOp.hasZeroOffset() && extractSliceOp.hasUnitStride();
  }

  bool isIdentityInsertSliceOp(tensor::InsertSliceOp insertSliceOp) const {
    return tensor::isCastLikeInsertSliceOp(insertSliceOp) &&
           insertSliceOp.getSourceType() == insertSliceOp.getResultType() &&
           insertSliceOp.hasZeroOffset() && insertSliceOp.hasUnitStride();
  }

  Value getIdentitySliceSource(Value value) const {
    if (auto extractSliceOp = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      if (isIdentityExtractSliceOp(extractSliceOp)) {
        return extractSliceOp.getSource();
      }
      return {};
    }
    if (auto insertSliceOp = value.getDefiningOp<tensor::InsertSliceOp>()) {
      if (isIdentityInsertSliceOp(insertSliceOp)) {
        return insertSliceOp.getSource();
      }
      return {};
    }
    return {};
  }
};

/// Folds a transfer_read that reads from the result of a transfer_write on
/// the same region (Read-After-Write) into arithmetic on the written value,
/// the original tensor, the masks, and the read's padding.
///
/// The general semantics are:
///
///   written_tensor[i] = wMask[i] ? valToStore[i] : original[i]
///   result[i]         = rMask[i] ? written_tensor[i] : rPad
///
/// Which gives:
///   result = select(rMask, select(wMask, valToStore, original),
///   broadcast(rPad))
///
/// Special cases avoid emitting unnecessary IR:
///   - No wMask (unmasked write): wMask is implicitly all-true, inner select
///     collapses to valToStore.
///   - No rMask (unmasked read): rMask is implicitly all-true, outer select
///     collapses away.
///   - wMask == rMask: the original tensor is never needed (anywhere rMask is
///     true, wMask is also true), so the inner select collapses to valToStore.
///
/// After bufferization, this generally removes the need for materializing the
/// write to memory.
// TODO: Consider upstreaming
struct FoldTransferRAW : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (!readOp.hasPureTensorSemantics()) {
      return failure();
    }

    auto writeOp = dyn_cast_if_present<vector::TransferWriteOp>(
        readOp.getBase().getDefiningOp());
    if (!writeOp || !writeOp.hasPureTensorSemantics()) {
      return failure();
    }

    Value valToStore = writeOp.getValueToStore();
    if (valToStore.getType() != readOp.getType()) {
      return failure();
    }

    // Work only with trivial or equal indices.
    if ((llvm::any_of(readOp.getIndices(),
                      [](Value v) { return !isZeroInteger(v); }) ||
         llvm::any_of(writeOp.getIndices(),
                      [](Value v) { return !isZeroInteger(v); })) &&
        (readOp.getIndices() != writeOp.getIndices())) {
      return failure();
    }

    if (!readOp.getPermutationMap().isMinorIdentity() ||
        !writeOp.getPermutationMap().isMinorIdentity()) {
      return failure();
    }

    TypedValue<VectorType> wMask = writeOp.getMask();
    TypedValue<VectorType> rMask = readOp.getMask();

    // If at least one operation claims that the position is in-bounds
    // then we can fold.
    //
    // Case 1.1: w_ib = false, r_ib = true, position is actually in_bounds
    // We write val, we read val, we can fold RAW to val.
    // Case 1.2: w_ib = false, r_ib = true, position is NOT in_bounds
    // We skip write, read says it is in_bounds, but that is false, which is UB
    // therefore we can fold to val.
    // Case 2.1: w_ib = true, r_ib = false, position is actually in_bounds
    // We write val, we read val, we can fold RAW to val.
    // Case 2.2: w_ib = true, r_ib = false, position is NOT in_bounds
    // UB on the write, therefore we can fold.
    if (readOp.hasOutOfBoundsDim() && writeOp.hasOutOfBoundsDim()) {
      return failure();
    }

    // Build the inner value: select(wMask, valToStore, original).
    // When wMask is absent (unmasked write) or wMask == rMask (original is
    // never accessed), this simplifies to just valToStore.
    Value inner = valToStore;
    bool needsOriginal = wMask && !areEquivalentMasks(wMask, rMask);
    if (needsOriginal) {
      Value originalRead = vector::TransferReadOp::create(
          rewriter, readOp.getLoc(), readOp.getType(), writeOp.getBase(),
          readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(),
          /*mask=*/Value(), readOp.getInBoundsAttr());
      inner = arith::SelectOp::create(rewriter, readOp.getLoc(), wMask,
                                      valToStore, originalRead);
    }

    if (!rMask) {
      rewriter.replaceOp(readOp, inner);
      return success();
    }

    // Build the outer value: select(rMask, inner, broadcast(rPad)).
    // When rMask is absent (unmasked read), the result is just inner.
    Value rPad = readOp.getPadding();
    assert(!isa<VectorType>(rPad.getType()) &&
           "masked transfers on vector element types are not supported; see "
           "verifyTransferOp in upstream MLIR VectorOps.cpp");
    Value padVal = vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                               valToStore.getType(), rPad);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(readOp, rMask, inner, padVal);
    return success();
  }
};

/// Folds transfer_read(tensor.empty).
///
/// Since tensor.empty has unspecified contents, reading from it produces
/// an unspecified value, which is exactly the semantics of ub.poison.
/// Out of bounds means that pad is used.
///
///   Case 1 — fully in-bounds, no mask:
///     %e = tensor.empty() : tensor<128xf16>
///     %r = vector.transfer_read %e[%c0], %pad {in_bounds = [true]}
///   ->
///     %r = ub.poison : vector<128xf16>
///
///   Case 2 — fully in-bounds, masked:
///     %e = tensor.empty() : tensor<128xf16>
///     %r = vector.transfer_read %e[%c0], %pad, %mask {in_bounds = [true]}
///   ->
///     %poison = ub.poison : vector<128xf16>
///     %bcast  = vector.broadcast %pad : f16 to vector<128xf16>
///     %r = arith.select %mask, %poison, %bcast
///
///   Case 3 — not fully in-bounds, no mask:
///     %e = tensor.empty() : tensor<100xf16>
///     %r = vector.transfer_read %e[%c0], %pad
///                              : tensor<100xf16>, vector<128xf16>
///   ->
///     %r = vector.broadcast %pad : f16 to vector<128xf16>
///
///   Case 4 — not fully in-bounds, masked:
///     %e = tensor.empty() : tensor<100xf16>
///     %r = vector.transfer_read %e[%c0], %pad, %mask
///                              : tensor<100xf16>, vector<128xf16>
///   ->
///     %r = vector.broadcast %pad : f16 to vector<128xf16>
struct FoldTransferReadOfEmptyTensor
    : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (!op.getBase().getDefiningOp<tensor::EmptyOp>()) {
      return failure();
    }

    if (!op.getPermutationMap().isMinorIdentity()) {
      return failure();
    }

    bool fullyInBounds =
        llvm::all_of(op.getInBoundsValues(), [](bool v) { return v; });
    TypedValue<VectorType> mask = op.getMask();

    if (mask && fullyInBounds) {
      // Masked, fully in-bounds: mask-on lanes read unspecified contents
      // (poison), mask-off lanes produce the padding value.
      Value rPad = op.getPadding();
      assert(!isa<VectorType>(rPad.getType()) &&
             "masked transfers on vector element types are not supported; "
             "see verifyTransferOp in upstream MLIR VectorOps.cpp");
      Value poison = ub::PoisonOp::create(rewriter, op.getLoc(), op.getType());
      Value padVal = vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                                 op.getType(), rPad);
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, poison, padVal);
      return success();
    }

    if (!mask && fullyInBounds) {
      // Unmasked, fully in-bounds: every lane reads unspecified contents.
      rewriter.replaceOp(
          op, ub::PoisonOp::create(rewriter, op.getLoc(), op.getType()));
      return success();
    }

    // Not fully in-bounds (with or without mask): out-of-bounds lanes
    // produce pad, and in-bounds lanes read unspecified contents from
    // tensor.empty, so we may choose pad for those too.
    Value rPad = op.getPadding();
    rewriter.replaceOp(op, vector::BroadcastOp::create(rewriter, rPad.getLoc(),
                                                       op.getType(), rPad));
    return success();
  }
};
} // namespace

static LogicalResult
canonicalizeLoopCarriedTransferState(mlir::FunctionOpInterface funcOp,
                                     MLIRContext *context) {
  // Keep dim/shape cleanup separate from transfer cleanup. FoldTransferRAW is
  // not reversible: if it runs before masks based on loop-carried tensor dims
  // are normalized to the loop init shape, it may conservatively materialize an
  // extra read of the original tensor and block later iter_arg promotion.
  {
    RewritePatternSet patterns(context);
    patterns.add<FoldDimOfShapePreservingTransferIterArg>(context);
    tensor::DimOp::getCanonicalizationPatterns(patterns, context);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }
  {
    RewritePatternSet patterns(context);
    patterns.add<FoldTransferWriteChain, FoldTransferRAW,
                 FoldTransferReadOfEmptyTensor>(context);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult
canonicalizeAndHoistLoopCarriedTransferState(mlir::FunctionOpInterface funcOp,
                                             MLIRContext *context) {
  if (failed(canonicalizeLoopCarriedTransferState(funcOp, context))) {
    return failure();
  }
  moveLoopInvariantCodeFromGuaranteedLoops(funcOp);

  RewritePatternSet hoistingPatterns(context);
  hoistingPatterns.add<MarkHoistableLoopCarriedTransferIterArg>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(hoistingPatterns)))) {
    return failure();
  }
  return IREE::Util::eliminateHoistableConversions(funcOp);
}

// Find the earliest insertion point in the block for the given operation.
static Operation *getEarliestInsertionPointInsideBlock(Block *block,
                                                       Operation *op) {

  Operation *currInsertionPoint = &(*block->getOperations().begin());
  DominanceInfo dominanceInfo(currInsertionPoint);

  for (auto operand : op->getOperands()) {
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      continue;
    }
    Operation *defOp = operand.getDefiningOp();
    if (!dominanceInfo.dominates(defOp, currInsertionPoint)) {
      currInsertionPoint = defOp;
    }
  }
  return currInsertionPoint;
}

void OptimizeTensorInsertExtractSlicesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());
  MLIRContext *context = &getContext();

  // TODO: This is a temporary hack enabled for bufferization to
  // get rid of empty buffers.
  // Tracked here: https://github.com/llvm/llvm-project/issues/122869
  funcOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    Block *currBlock = extractSliceOp.getOperation()->getBlock();
    auto latestInsertionPoint =
        getEarliestInsertionPointInsideBlock(currBlock, extractSliceOp);
    extractSliceOp->moveAfter(latestInsertionPoint);
  });

  // Run the vector transfer-specific hoisting before generic subset hoisting
  // so masked transfer loops get the required per-iteration padding semantics.
  if (failed(canonicalizeAndHoistLoopCarriedTransferState(funcOp, context))) {
    return signalPassFailure();
  }
  LDBG() << "after early hoisting loop carried vector transfers" << funcOp;

  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  LDBG() << "after hoisting loop invariant subsets\n" << funcOp;

  funcOp.walk([&](scf::ForOp forOp) {
    hoistSubsetWithLoopInvariantTensor(rewriter, forOp);
  });
  LDBG() << "after hoisting subset loop invariant tensors" << funcOp;

  if (failed(canonicalizeAndHoistLoopCarriedTransferState(funcOp, context))) {
    return signalPassFailure();
  }
  LDBG() << "after late hoisting loop carried vector transfers" << funcOp;

  RewritePatternSet patterns(context);
  populateVectorTransferTensorSliceTransforms(patterns);
  // Keep the final hoisting marker in this general cleanup because its
  // transfer/slice folds can expose additional loop-carried transfers.
  patterns.add<MarkHoistableLoopCarriedTransferIterArg>(context);
  scf::ForOp::getCanonicalizationPatterns(patterns, context);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
  // Run after subset hoisting so retargeting reads through identity slices
  // does not create direct tensor iter_arg uses that block subset matching.
  patterns.add<FoldTransferReadOfIdentitySlice>(context);
  if (foldIdentitySlices) {
    patterns.add<CastLikeExtractSliceOpFolder>(context);
    patterns.add<CastLikeInsertSliceOpFolder>(context);
  }
  // Apply masked transfer_write + transfer_read folding to avoid spurious
  // (future) roundtrips to memory.
  // TODO: consider upstreaming.
  patterns.add<FoldTransferRAW, FoldTransferReadOfEmptyTensor>(context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
  if (failed(IREE::Util::eliminateHoistableConversions(funcOp))) {
    return signalPassFailure();
  }

  LDBG() << "after folding tensor.extract_slice and vector.transfer_read Ops \n"
         << funcOp;
}

} // namespace
} // namespace mlir::iree_compiler
