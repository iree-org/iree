// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/TensorMaskingOpInterface.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

#include "iree/compiler/Codegen/Interfaces/TensorMaskingOpInterface.cpp.inc" // IWYU pragma: keep

namespace {

static bool isIterationCarriedArg(OpOperand &arg, linalg::GenericOp op) {
  auto bbArg = dyn_cast<BlockArgument>(arg.get());
  if (!bbArg) {
    return false;
  }
  OpOperand *inputOperand = op.getMatchingOpOperand(bbArg);
  return op.isDpsInit(inputOperand);
}

/// Returns true if the operation needs to be padded to align its iteration
/// domain to the given `padMultiples`. Padding is needed when:
///   - A dimension size is dynamic (not statically known), or
///   - A dimension size is not evenly divisible by its corresponding multiple.
/// A pad multiple of 0 indicates that no padding is required for that
/// dimension.
static bool isPaddingNeeded(OpBuilder &b, TilingInterface op,
                            ArrayRef<OpFoldResult> padMultiples) {
  SmallVector<Range> iterationDomain = op.getIterationDomain(b);

  // Mismatched sizes. Return true (assume padding is needed) to allow later
  // validation stages to emit proper diagnostics.
  if (iterationDomain.size() < padMultiples.size()) {
    return true;
  }

  for (size_t i = 0; i < padMultiples.size(); ++i) {
    std::optional<int64_t> padSize = getConstantIntValue(padMultiples[i]);
    if (!padSize || *padSize == 0) {
      continue;
    }
    std::optional<int64_t> dimSize =
        getConstantIntValue(iterationDomain[i].size);
    if (!dimSize) {
      // Dynamic dimension - assume padding is needed.
      return true;
    }
    if (*dimSize % *padSize != 0) {
      return true;
    }
  }
  return false;
}

static FailureOr<linalg::PadTilingInterfaceResult>
maskTilingInterfaceOp(OpBuilder &b, Operation *op,
                      ArrayRef<OpFoldResult> padMultiples) {
  auto poison = ub::PoisonAttr::get(b.getContext());
  SmallVector<Attribute> padValues(op->getNumOperands(), poison);
  TilingInterface tilingInterfaceOp = cast<TilingInterface>(op);
  // Set options.
  linalg::PadTilingInterfaceOptions options =
      linalg::PadTilingInterfaceOptions()
          .setPaddingSizes(padMultiples)
          .setPaddingValues(padValues)
          .setPadToMultipleOf(true);

  FailureOr<linalg::PadTilingInterfaceResult> result =
      linalg::rewriteAsPaddedOp(b, tilingInterfaceOp, options);
  return result;
}

static SmallVector<OpFoldResult>
getBoundsToGuard(OpBuilder &b, Operation *unpaddedOp, Operation *paddedOp) {
  OpBuilder::InsertionGuard guard(b);
  auto unpaddedIface = cast<TilingInterface>(unpaddedOp);
  auto paddedIface = cast<TilingInterface>(paddedOp);
  b.setInsertionPoint(unpaddedOp);
  SmallVector<Range> oldItSpace = unpaddedIface.getIterationDomain(b);
  b.setInsertionPoint(paddedOp);
  SmallVector<Range> newItSpace = paddedIface.getIterationDomain(b);
  SmallVector<OpFoldResult> oldDims =
      llvm::map_to_vector(oldItSpace, [](Range r) { return r.size; });
  SmallVector<OpFoldResult> newDims =
      llvm::map_to_vector(newItSpace, [](Range r) { return r.size; });
  for (auto [oldDim, newDim, itType] : llvm::zip_equal(
           oldDims, newDims, unpaddedIface.getLoopIteratorTypes())) {
    // Non-reduction dimensions don't need to be guarded, as they will be
    // guarded at the user of the mask.
    if (!linalg::isReductionIterator(itType)) {
      oldDim = OpFoldResult();
      continue;
    }
    // If both the old and new dim are equal, no need to guard.
    if (oldDim == newDim) {
      oldDim = OpFoldResult();
      continue;
    }
  }
  return oldDims;
}

/// For a dimension to be in bounds in the linalg op's body:
///
///  %is_in_bounds(%dim) = linalg.index %dim < tensor.dim %dim (of the old
///  op)
///
/// With this, we can check if the reduction needs to use a neutral element
/// by checking if the current iteration is in bounds for all reduction
/// dimensions.
static OpFoldResult isGenericIterationInBounds(OpBuilder &b, Location loc,
                                               ArrayRef<OpFoldResult> bounds) {
  OpFoldResult isReductionInBounds = b.getBoolAttr(true);
  for (auto [index, bound] : llvm::enumerate(bounds)) {
    if (!bound) {
      continue;
    }
    Value loopIdx = linalg::IndexOp::create(b, loc, index);
    Value isInBounds = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, loopIdx,
        getValueOrCreateConstantIndexOp(b, loc, bounds[index]));
    isReductionInBounds = b.createOrFold<arith::AndIOp>(
        loc, getValueOrCreateConstantIntOp(b, loc, isReductionInBounds),
        isInBounds);
  }
  return isReductionInBounds;
}

/// Given a masked tensor operand indexed in the operation using `map` of an
/// operation with bounds `bounds`, select the neutral element when the operand
///  out of bounds. If a bound for a dim is null, assume it is in bounds.
static void selectNeutralElementForReducedDims(OpBuilder &b, OpOperand &operand,
                                               arith::AtomicRMWKind kind,
                                               ArrayRef<OpFoldResult> bounds,
                                               AffineMap map) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(operand.getOwner());
  Location loc = operand.getOwner()->getLoc();
  // Get the neutral element for masking.
  Value neutralEl = arith::getIdentityValue(
      kind, getElementTypeOrSelf(operand.get().getType()), b, loc,
      /*useOnlyFiniteValue=*/false);
  // Compute operand bounds from iteration space bounds.
  SmallVector<OpFoldResult> operandBounds = applyPermutationMap(map, bounds);
  AffineMap elwiseMap =
      AffineMap::getMultiDimIdentityMap(operandBounds.size(), b.getContext());
  SmallVector<OpFoldResult> operandDims =
      tensor::getMixedSizes(b, loc, operand.get());
  auto emptyOp =
      tensor::EmptyOp::create(b, loc, operandDims, neutralEl.getType());
  Type operandTy = emptyOp.getType();
  // Create a elementwise linalg.generic to select the neutral element.
  auto genericOp = linalg::GenericOp::create(
      b, loc, {operandTy}, {operand.get()}, {emptyOp.getResult()},
      {elwiseMap, elwiseMap},
      SmallVector<utils::IteratorType>(operandBounds.size(),
                                       utils::IteratorType::parallel),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Check if the loop iterations go out of bounds.
        OpFoldResult inBounds =
            isGenericIterationInBounds(b, loc, operandBounds);
        Value selected = b.createOrFold<arith::SelectOp>(
            loc, getValueOrCreateConstantIntOp(b, loc, inBounds), args[0],
            neutralEl);
        linalg::YieldOp::create(b, loc, selected);
      });
  operand.set(genericOp.getResult(0));
}

struct LinalgGenericOpInterface final
    : TensorMaskingOpInterface::ExternalModel<LinalgGenericOpInterface,
                                              linalg::GenericOp> {

  FailureOr<SmallVector<Value>>
  getMaskedImplementation(Operation *op, OpBuilder &builder,
                          ArrayRef<OpFoldResult> padMultiples) const {
    FailureOr<linalg::PadTilingInterfaceResult> result =
        maskTilingInterfaceOp(builder, op, padMultiples);
    // Properly mask the reductions to avoid reducing poison values.
    auto paddedGeneric =
        cast<linalg::GenericOp>(result->paddedOp.getOperation());
    // TODO: Add special handling for Contraction/Convolution with no selects in
    // the body.
    auto linalgOp = cast<linalg::GenericOp>(op);
    if (failed(maskReductionsInGenericBody(builder, linalgOp, paddedGeneric))) {
      return failure();
    }
    return result->replacements;
  }

private:
  /// Mask the poison inputs for reductions by selecting the neutral element on
  /// the masked out-of-bounds iterations.
  LogicalResult
  maskReductionsInGenericBody(OpBuilder &builder, linalg::GenericOp oldLinalgOp,
                              linalg::GenericOp paddedLinalgOp) const {
    Location loc = paddedLinalgOp.getLoc();
    builder.setInsertionPointToStart(paddedLinalgOp.getBody());
    SmallVector<OpFoldResult> oldBounds =
        getBoundsToGuard(builder, oldLinalgOp, paddedLinalgOp);
    OpFoldResult isReductionInBounds =
        isGenericIterationInBounds(builder, loc, oldBounds);

    SmallVector<Operation *> reductions;
    for (auto [index, initOpOperand] :
         llvm::enumerate(paddedLinalgOp.getDpsInitsMutable())) {
      SmallVector<Operation *> combinerOps;
      matchReduction(paddedLinalgOp.getRegionOutputArgs(), index, combinerOps);
      reductions.insert(reductions.begin(), combinerOps.begin(),
                        combinerOps.end());
    }

    for (Operation *reduction : reductions) {
      std::optional<TypedAttr> reductionIdentity =
          arith::getNeutralElement(reduction);
      if (!reductionIdentity.has_value()) {
        paddedLinalgOp.emitError("failed to get neutral element for reduction");
        return failure();
      }
      // Mask out inputs on out-of-bounds iterations.
      builder.setInsertionPoint(reduction);
      for (OpOperand &opOperand : reduction->getOpOperands()) {
        if (!isIterationCarriedArg(opOperand, paddedLinalgOp)) {
          Value neutralElValue = arith::ConstantOp::create(
              builder, loc, reductionIdentity.value());
          Value selected = builder.createOrFold<arith::SelectOp>(
              loc,
              getValueOrCreateConstantIntOp(builder, loc, isReductionInBounds),
              opOperand.get(), neutralElValue);
          opOperand.set(selected);
        }
      }
    }

    return success();
  }
};

struct OnlineAttentionOpInterface final
    : TensorMaskingOpInterface::ExternalModel<
          OnlineAttentionOpInterface, IREE::LinalgExt::OnlineAttentionOp> {

  FailureOr<SmallVector<Value>>
  getMaskedImplementation(Operation *op, OpBuilder &builder,
                          ArrayRef<OpFoldResult> padMultiples) const {
    auto onlineAttentionOp = cast<IREE::LinalgExt::OnlineAttentionOp>(op);

    // Check if padding is actually needed. If all dimensions are already
    // aligned to the pad multiples, we can skip padding entirely.
    auto tilingOp = cast<TilingInterface>(op);
    if (!isPaddingNeeded(builder, tilingOp, padMultiples)) {
      return failure();
    }

    if (!onlineAttentionOp.getMask()) {
      op->emitError(
          "Padding OnlineAttention without existing mask is not yet supported");
      return failure();
    }

    FailureOr<linalg::PadTilingInterfaceResult> result =
        maskTilingInterfaceOp(builder, op, padMultiples);
    if (failed(result)) {
      op->emitError("failed to pad op");
      return failure();
    }
    // Select neutral values for masked out-of-bounds iterations.
    auto paddedOp = cast<IREE::LinalgExt::OnlineAttentionOp>(result->paddedOp);
    SmallVector<OpFoldResult> bounds = getBoundsToGuard(builder, op, paddedOp);

    selectNeutralElementForReducedDims(builder, paddedOp.getQueryMutable(),
                                       arith::AtomicRMWKind::addf, bounds,
                                       paddedOp.getQueryMap());
    selectNeutralElementForReducedDims(builder, paddedOp.getKeyMutable(),
                                       arith::AtomicRMWKind::addf, bounds,
                                       paddedOp.getKeyMap());
    selectNeutralElementForReducedDims(builder, paddedOp.getValueMutable(),
                                       arith::AtomicRMWKind::addf, bounds,
                                       paddedOp.getValueMap());
    if (paddedOp.getMask()) {
      selectNeutralElementForReducedDims(builder, paddedOp.getMaskMutable()[0],
                                         arith::AtomicRMWKind::maximumf, bounds,
                                         *paddedOp.getMaskMap());
    }
    return result->replacements;
  }
};

} // namespace

void registerTensorMaskingOpInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::GenericOp::attachInterface<LinalgGenericOpInterface>(*ctx);
  });
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        IREE::LinalgExt::OnlineAttentionOp::attachInterface<
            OnlineAttentionOpInterface>(*ctx);
      });
}

}; // namespace mlir::iree_compiler
