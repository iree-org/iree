// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_VMVXSPLITREDUCTIONPASS
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"

namespace {

static bool isSupportedSum(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 ||
      op.getNumDpsInits() != 1 || op.getNumReductionLoops() != 1 ||
      op.hasIndexSemantics() || op.getLibraryCallAttr()) {
    return false;
  }
  auto inputType = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto outputType = cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!inputType || !inputType.getElementType().isF32() ||
      !outputType.getElementType().isF32() || inputType.getEncoding() ||
      outputType.getEncoding()) {
    return false;
  }
  auto maps = op.getIndexingMapsArray();
  if (!maps[0].isIdentity() ||
      maps[1] !=
          AffineMap::getMultiDimIdentityMap(op.getNumLoops(), op.getContext())
              .dropResult(op.getNumLoops() - 1) ||
      op.getIteratorTypesArray().back() != utils::IteratorType::reduction) {
    return false;
  }
  // Distribution has already consumed this configuration in the VMVX pipeline.
  // Other annotations may carry contracts the new iteration space cannot
  // retain.
  if (llvm::any_of(op->getDiscardableAttrs(), [](NamedAttribute attr) {
        return attr.getName() != kConfigAttrName;
      })) {
    return false;
  }
  return llvm::all_of(op.getBody()->getOperations(), [](Operation &nested) {
    return nested.getNumRegions() == 0 && isMemoryEffectFree(&nested);
  });
}

struct ReductionSplitPlan {
  int64_t chunkCount;
  TypedAttr identity;
  arith::AddFOp combiner;
  Value contribution;
  Value initialValue;
};

static std::optional<ReductionSplitPlan>
calculateReductionSplit(linalg::GenericOp op, int64_t chunkSize) {
  if (!isSupportedSum(op)) {
    return std::nullopt;
  }
  auto inputType = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  int64_t extent = inputType.getShape().back();
  if (ShapedType::isDynamic(extent) || extent <= chunkSize ||
      extent % chunkSize != 0) {
    return std::nullopt;
  }
  SmallVector<Operation *> combiners;
  Value contribution = matchReduction(op.getRegionOutputArgs(), 0, combiners);
  if (!contribution || combiners.size() != 1) {
    return std::nullopt;
  }
  auto add = dyn_cast<arith::AddFOp>(combiners.front());
  if (!add) {
    return std::nullopt;
  }
  // Positive zero would erase the sign of a partial sum containing only -0.
  auto identity = FloatAttr::get(
      Float32Type::get(op.getContext()),
      llvm::APFloat::getZero(llvm::APFloat::IEEEsingle(), /*Negative=*/true));
  return ReductionSplitPlan{extent / chunkSize, identity, add, contribution,
                            op.getRegionOutputArgs()[0]};
}

// sum(input[M, N], init[M]) becomes
// sum(sum(reshape(input)[M, N/K, K], -0), init[M]).
// Each partial accumulates at most K adjacent contributions. Supply dynamic
// parallel extents explicitly, and use the caller init only in the final sum.
static linalg::GenericOp splitSumReduction(RewriterBase &rewriter,
                                           linalg::GenericOp op,
                                           int64_t chunkSize,
                                           const ReductionSplitPlan &plan) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value input = op.getDpsInputs()[0];
  Value init = op.getDpsInits()[0];
  SmallVector<OpFoldResult> expandedShape =
      tensor::getMixedSizes(rewriter, loc, input);
  expandedShape.back() = rewriter.getIndexAttr(plan.chunkCount);
  expandedShape.push_back(rewriter.getIndexAttr(chunkSize));
  SmallVector<int64_t> staticShape;
  for (OpFoldResult dim : expandedShape) {
    staticShape.push_back(
        getConstantIntValue(dim).value_or(ShapedType::kDynamic));
  }
  SmallVector<ReassociationIndices> reassociation;
  int64_t inputRank = cast<RankedTensorType>(input.getType()).getRank();
  for (int64_t dim = 0; dim < inputRank - 1; ++dim) {
    reassociation.push_back({dim});
  }
  reassociation.push_back({inputRank - 1, inputRank});
  auto expandedType = RankedTensorType::get(staticShape, rewriter.getF32Type());
  Value expanded = tensor::ExpandShapeOp::create(
      rewriter, loc, expandedType, input, reassociation, expandedShape);

  SmallVector<OpFoldResult> partialShape =
      tensor::getMixedSizes(rewriter, loc, init);
  partialShape.push_back(rewriter.getIndexAttr(plan.chunkCount));
  Value empty = tensor::EmptyOp::create(rewriter, loc, partialShape,
                                        rewriter.getF32Type());
  Value identity = arith::ConstantOp::create(rewriter, loc, plan.identity);
  Value partialInit =
      linalg::FillOp::create(rewriter, loc, identity, empty).getResult(0);
  AffineMap partialInputMap = rewriter.getMultiDimIdentityMap(inputRank + 1);
  AffineMap partialOutputMap = partialInputMap.dropResult(inputRank);
  SmallVector<utils::IteratorType> partialIterators(
      inputRank, utils::IteratorType::parallel);
  partialIterators.push_back(utils::IteratorType::reduction);
  auto partial = linalg::GenericOp::create(
      rewriter, loc, TypeRange{empty.getType()}, ValueRange{expanded},
      ValueRange{partialInit},
      ArrayRef<AffineMap>{partialInputMap, partialOutputMap}, partialIterators);
  rewriter.inlineRegionBefore(op.getRegion(), partial.getRegion(),
                              partial.getRegion().begin());
  auto final = linalg::GenericOp::create(
      rewriter, loc, op->getResultTypes(), ValueRange{partial.getResult(0)},
      ValueRange{init}, op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      [plan](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        IRMapping mapping;
        mapping.map(plan.contribution, args[0]);
        mapping.map(plan.initialValue, args[1]);
        Operation *add = builder.clone(*plan.combiner, mapping);
        linalg::YieldOp::create(builder, nestedLoc, add->getResult(0));
      });
  if (auto config = getLoweringConfig(op)) {
    rewriter.modifyOpInPlace(final, [&] { setLoweringConfig(final, config); });
  }
  rewriter.replaceOp(op, final.getResults());
  return final;
}

class VMVXSplitReductionPass final
    : public impl::VMVXSplitReductionPassBase<VMVXSplitReductionPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (reductionChunkSize < 2) {
      getOperation()->emitError("reduction-chunk-size must be at least 2");
      return signalPassFailure();
    }
    if (!reassociateFpReductions) {
      return markAllAnalysesPreserved();
    }
    SmallVector<linalg::GenericOp> candidates;
    getOperation()->walk(
        [&candidates](linalg::GenericOp op) { candidates.push_back(op); });
    IRRewriter rewriter(&getContext());
    bool changed = false;
    for (linalg::GenericOp op : candidates) {
      while (auto plan = calculateReductionSplit(op, reductionChunkSize)) {
        op = splitSumReduction(rewriter, op, reductionChunkSize, *plan);
        changed = true;
        // The remaining reduction extent decreases by a factor of at least two.
      }
    }
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
