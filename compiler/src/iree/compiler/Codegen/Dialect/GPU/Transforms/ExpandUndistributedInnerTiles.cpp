// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_EXPANDUNDISTRIBUTEDINNERTILESPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct ExpandUndistributedInnerTilesPass final
    : impl::ExpandUndistributedInnerTilesPassBase<
          ExpandUndistributedInnerTilesPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static LogicalResult materializeOperandExpandedShape(
    OpBuilder &builder, Codegen::InnerTiledOp tiledOp, int64_t operandIndex,
    VectorType unexpandedInnerLogicalType,
    std::optional<ArrayRef<int64_t>> permutation,
    SmallVectorImpl<ReassociationIndices> &reassociations,
    RankedTensorType &resultType) {

  Codegen::InnerTileDescAttrInterface tiledOpKind = tiledOp.getKind();
  Value operand = tiledOp.getOperand(operandIndex);
  auto operandType = cast<RankedTensorType>(operand.getType());
  ArrayRef<int64_t> operandShape = operandType.getShape();
  int64_t outerRank = tiledOp.getOperandOuterRank(operandIndex);
  int64_t innerRank = operandType.getRank() - outerRank;

  SmallVector<int64_t> unexpandedSizes(unexpandedInnerLogicalType.getShape());
  if (permutation.has_value()) {
    // Guard against re-running this pattern or other unusial states
    if (permutation->size() != unexpandedSizes.size()) {
      return failure();
    }
    applyPermutationToVector(unexpandedSizes, *permutation);
  }
  // If we're not in the exact shape that needs expanding, don't expand.
  if (operandShape.drop_front(outerRank) !=
      ArrayRef<int64_t>{unexpandedSizes}) {
    return failure();
  }

  SmallVector<int64_t> resultShape(operandShape.drop_back(innerRank));
  SmallVector<ReassociationIndices> reInds = llvm::map_to_vector(
      llvm::seq<int64_t>(outerRank), [](int64_t idx) -> ReassociationIndices {
        return ReassociationIndices({idx});
      });
  int64_t reAssocIdx = outerRank;
  bool isNonTrivial = false;
  for (int64_t innerDim : llvm::seq(innerRank)) {
    int64_t logicalInnerDim = innerDim;
    if (permutation.has_value()) {
      // The permutation semantics are that permutation[i] is the index in the
      // logical (for matmul, this is row-major) layout of the intrinsic that
      // corresponds to inner tile dimension i. That is if your logical shape is
      // K0 x Kb x N and the permutation is [2, 0, 1], the actual shape of the
      // tile will be N x Ko x Kb. Therefore, we can go from the "physical"
      // index to the "logical" one (which is needed for looking up any needed
      // reshapes) with a simple array lookup.
      logicalInnerDim = (*permutation)[innerDim];
    }
    std::optional<SmallVector<int64_t, 2>> maybeExpansion =
        tiledOpKind.getUndistributedTileDimExpansion(operandIndex,
                                                     logicalInnerDim);
    if (!maybeExpansion) {
      resultShape.push_back(operandShape[outerRank + innerDim]);
      reInds.push_back(ReassociationIndices{reAssocIdx++});
      continue;
    }
    int64_t expandRank = maybeExpansion->size();
    llvm::append_range(resultShape, *maybeExpansion);
    reInds.push_back(
        llvm::to_vector(llvm::seq(reAssocIdx, reAssocIdx + expandRank)));
    reAssocIdx += expandRank;
    isNonTrivial = true;
  }
  if (!isNonTrivial) {
    // Three's nothing to expand. Note that this is always the case for
    // DataTiledMMAAttr and VirtualMMAAttr.
    return failure();
  }

  reassociations = reInds;
  resultType = operandType.clone(resultShape);
  return success();
}

namespace {
struct ExpandInnerTileShapes final : OpRewritePattern<Codegen::InnerTiledOp> {
  using OpRewritePattern::OpRewritePattern;

  ExpandInnerTileShapes(MLIRContext *context, bool expandInputs,
                        bool expandOutputs)
      : OpRewritePattern<Codegen::InnerTiledOp>(context),
        expandInputs(expandInputs), expandOutputs(expandOutputs) {}

  LogicalResult matchAndRewrite(Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (!tiledOp.hasTensorSemantics()) {
      return failure();
    }
    Location loc = tiledOp.getLoc();

    int64_t numInputs = tiledOp.getNumInputs();
    int64_t numOperands = tiledOp.getNumOperands();
    int64_t firstOperand = expandInputs ? 0 : numInputs;
    int64_t lastOperand = expandOutputs ? numOperands : numInputs;

    std::optional<ArrayAttr> permutationsAttr = tiledOp.getPermutations();

    SmallVector<tensor::ExpandShapeOp> maybeExpands(numOperands, nullptr);
    SmallVector<Value> newOperands(tiledOp.getOperands());
    SmallVector<Attribute> newPermutations;
    if (permutationsAttr.has_value() && *permutationsAttr) {
      newPermutations = llvm::to_vector(*permutationsAttr);
    }

    SmallVector<VectorType> unexpandedLogicalTypes;
    tiledOp.getKind().getUndistributedTileTypes(unexpandedLogicalTypes);
    for (int64_t opIndex : llvm::seq(firstOperand, lastOperand)) {
      Value operand = newOperands[opIndex];
      std::optional<ArrayRef<int64_t>> permutation;
      if (permutationsAttr.has_value() && *permutationsAttr) {
        permutation =
            cast<DenseI64ArrayAttr>((*permutationsAttr)[opIndex]).asArrayRef();
      }

      SmallVector<ReassociationIndices> reassociations;
      RankedTensorType expandedType;
      // Note: failure here includes being trivial.
      if (failed(materializeOperandExpandedShape(
              rewriter, tiledOp, opIndex, unexpandedLogicalTypes[opIndex],
              permutation, reassociations, expandedType))) {
        continue;
      }
      auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
          loc, expandedType, operand, reassociations);
      maybeExpands[opIndex] = expandOp;
      newOperands[opIndex] = expandOp.getResult();

      // The existing permutation does logical -> physical tile remapping,
      // so to create a new one, we take the reassociations of the
      // physical tiles and permute them. Then, if you read these permuted
      // reassociation indices (and subtract off the outer rank), they'll
      // be indexes into the new physical dimensions in logical order.
      // Inverting again will give the permutation we want to use.
      if (permutation) {
        int64_t outerRank = tiledOp.getOperandOuterRank(opIndex);
        SmallVector<int64_t> invPermutation =
            invertPermutationVector(*permutation);
        SmallVector<ReassociationIndices> logicalReassocs = applyPermutation(
            ArrayRef<ReassociationIndices>{reassociations}.drop_front(
                outerRank),
            invPermutation);
        SmallVector<int64_t> newInvPermutation;
        for (auto reassoc : logicalReassocs) {
          llvm::append_range(newInvPermutation,
                             llvm::map_range(reassoc, [&](int64_t idx) {
                               return idx - outerRank;
                             }));
        }
        newPermutations[opIndex] = rewriter.getDenseI64ArrayAttr(
            invertPermutationVector(newInvPermutation));
      }
    }

    if (llvm::all_of(maybeExpands, [](auto e) { return e == nullptr; })) {
      // No expansions occurred, do nothing.
      return failure();
    }
    std::optional<ArrayAttr> newPermutationsAttr;
    if (permutationsAttr) {
      newPermutationsAttr = rewriter.getArrayAttr(newPermutations);
    }
    // Create the new inner_tiled op with the expanded type.
    auto expandedTiledOp = rewriter.create<Codegen::InnerTiledOp>(
        loc, /*inputs=*/ValueRange{newOperands}.take_front(numInputs),
        /*inits=*/ValueRange{newOperands}.drop_front(numInputs),
        tiledOp.getIndexingMaps(), tiledOp.getIteratorTypes(),
        tiledOp.getKind(), newPermutationsAttr);

    if (auto config = getLoweringConfig(tiledOp)) {
      setLoweringConfig(expandedTiledOp, config);
    }

    SmallVector<Value> newResults(expandedTiledOp.getResults());
    // If we had to expnad any accumulators, collapse the corresponding results
    // back to their original shape.
    for (auto [resIndex, result] : llvm::enumerate(newResults)) {
      tensor::ExpandShapeOp tiedInitExpand = maybeExpands[numInputs + resIndex];
      if (!tiedInitExpand) {
        continue;
      }
      result = rewriter.create<tensor::CollapseShapeOp>(
          loc, tiledOp.getResultTypes()[resIndex], result,
          tiedInitExpand.getReassociation());
    }

    rewriter.replaceOp(tiledOp, newResults);
    return success();
  }

private:
  bool expandInputs = true;
  bool expandOutputs = true;
};
} // namespace

void ExpandUndistributedInnerTilesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(context);
  patterns.insert<ExpandInnerTileShapes>(context, expandInputs, expandOutputs);
  walkAndApplyPatterns(funcOp, std::move(patterns));
}

} // namespace mlir::iree_compiler::IREE::GPU
