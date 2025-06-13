// Copyright 2024 The IREE Authors
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_CONCRETIZEMMASHAPESPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct ConcretizeMmaShapesPass final
    : impl::ConcretizeMmaShapesPassBase<ConcretizeMmaShapesPass> {
  using ConcretizeMmaShapesPassBase::ConcretizeMmaShapesPassBase;
  void runOnOperation() override;
};

LogicalResult materializeOperandConcreteShape(
    OpBuilder &builder, MMAAttr mma, IREE::GPU::MMAFragment fragment,
    Value operand, std::optional<ArrayRef<int64_t>> permutation,
    SmallVector<ReassociationIndices> &reassociations,
    RankedTensorType &resultType) {

  MMASingleSubgroupLayout layout = getSingleSubgroupLayout(mma, fragment);
  SmallVector<int64_t, 2> outerSizes = layout.outer;
  SmallVector<int64_t, 2> opaqueSizes;
  auto [m, n, k] = mma.getMNKShape();
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs: {
    opaqueSizes.append({m, k});
    break;
  }
  case IREE::GPU::MMAFragment::Rhs: {
    opaqueSizes.append({k, n});
    break;
  }
  case IREE::GPU::MMAFragment::Acc: {
    opaqueSizes.append({m, n});
    break;
  }
  }
  if (permutation.has_value()) {
    if (permutation.value().size() != outerSizes.size()) {
      return failure();
    }
    applyPermutationToVector(opaqueSizes, permutation.value());
    applyPermutationToVector(outerSizes, permutation.value());
  }

  // Inner tile must have sizes matching the opaque layout.
  auto operandType = llvm::cast<RankedTensorType>(operand.getType());
  ArrayRef<int64_t> operandShape = operandType.getShape();
  if (opaqueSizes != operandShape.take_back(opaqueSizes.size())) {
    return failure();
  }

  // Expand the shape of the inner tile to reflect the MMA thread layout.
  SmallVector<int64_t, 4> resultShape(operandShape.begin(),
                                      operandShape.end() - 2);
  SmallVector<ReassociationIndices> reInds =
      llvm::map_to_vector(llvm::seq<int64_t>(resultShape.size()),
                          [](int64_t idx) -> ReassociationIndices {
                            return ReassociationIndices({idx});
                          });
  int idx = reInds.size();
  for (auto [outer, native] : llvm::zip_equal(outerSizes, opaqueSizes)) {
    // Skip expansion if the outer dim is unit as the SingleSubgroupLayout gives
    // a guarantee that the |element| counts are contiguous within the layout,
    // and a unit outer implies a single offset and size for that dimension.
    if (outer == 1) {
      resultShape.push_back(native);
      reInds.push_back(ReassociationIndices({idx++}));
      continue;
    }

    // Reshape to [outer, native / outer] == [outer, thread * element]. This
    // corresponds to |outer| repetitions of the thread/element sublayout.
    resultShape.push_back(outer);
    assert(native % outer == 0 && "invalid mma layout");
    resultShape.push_back(native / outer);
    reInds.push_back(ReassociationIndices{idx, idx + 1});
    idx += 2;
  }

  reassociations = reInds;
  resultType = operandType.clone(resultShape);
  return success();
}

struct ConcretizeMmaOperandShape final
    : OpRewritePattern<Codegen::InnerTiledOp> {
  using OpRewritePattern::OpRewritePattern;

  ConcretizeMmaOperandShape(MLIRContext *context, MMAFragment fragment)
      : OpRewritePattern<Codegen::InnerTiledOp>(context), fragment(fragment) {}

  LogicalResult matchAndRewrite(Codegen::InnerTiledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    if (!mmaOp.hasTensorSemantics()) {
      return failure();
    }
    auto kind = dyn_cast<MMAAttr>(mmaOp.getKind());
    if (!kind) {
      return rewriter.notifyMatchFailure(
          mmaOp, "don't know how to concretize non-mma ops");
    }

    // Get the right operand and permutation for the `fragment`.
    SmallVector<Value> operands = mmaOp.getOperands();
    std::optional<ArrayAttr> permutationsAttr = mmaOp.getPermutations();
    int64_t opIndex;
    switch (fragment) {
    case MMAFragment::Lhs:
      opIndex = 0;
      break;
    case MMAFragment::Rhs:
      opIndex = 1;
      break;
    case MMAFragment::Acc:
      opIndex = 2;
      break;
    }

    Value operand = operands[opIndex];
    std::optional<ArrayRef<int64_t>> permutation;
    if (permutationsAttr) {
      permutation =
          cast<DenseI64ArrayAttr>((*permutationsAttr)[opIndex]).asArrayRef();
    }
    SmallVector<ReassociationIndices> reassociations;
    RankedTensorType concreteType;
    if (failed(materializeOperandConcreteShape(rewriter, kind, fragment,
                                               operand, permutation,
                                               reassociations, concreteType))) {
      return failure();
    }

    // Early exit if the operand is unaffected.
    if (llvm::all_of(reassociations, [](ReassociationIndices reassoc) {
          return reassoc.size() == 1;
        })) {
      return failure();
    }

    // Create the expand_shape.
    Location loc = mmaOp->getLoc();
    Value concreteOperand = rewriter
                                .create<tensor::ExpandShapeOp>(
                                    loc, concreteType, operand, reassociations)
                                .getResult();

    // Expand the permutation for the new inner dimensions of the expanded
    // inner_tiled operand.
    std::optional<ArrayAttr> newPermutationsAttr;
    if (permutationsAttr) {
      SmallVector<Attribute> newPermutations =
          llvm::to_vector(*permutationsAttr);
      int64_t outerRank = mmaOp.getOperandOuterRank(opIndex);
      SmallVector<ReassociationIndices> innerReInds(
          reassociations.begin() + outerRank, reassociations.end());
      for (auto &reInd : innerReInds) {
        for (auto &idx : reInd) {
          idx -= outerRank;
        }
      }
      // |perm| represents the permutation that takes the canonical (row major)
      // layout and converts it to the current layout. Take the inverse to
      // update to the expanded permutation and then invert back
      SmallVector<int64_t> invertPerm =
          invertPermutationVector(permutation.value());
      SmallVector<int64_t> expandedPerm;
      for (auto reInd : applyPermutation(innerReInds, invertPerm)) {
        expandedPerm.append(reInd);
      }
      expandedPerm = invertPermutationVector(expandedPerm);
      newPermutations[opIndex] = rewriter.getDenseI64ArrayAttr(expandedPerm);
      newPermutationsAttr = rewriter.getArrayAttr(newPermutations);
    }

    operands[opIndex] = concreteOperand;
    // Create the new inner_tiled op with the concrete type.
    auto concreteMmaOp = rewriter.create<Codegen::InnerTiledOp>(
        loc, /*inputs=*/ValueRange{operands}.drop_back(),
        /*inits=*/ValueRange{operands}.back(), mmaOp.getIndexingMaps(),
        mmaOp.getIteratorTypes(), mmaOp.getKind(), newPermutationsAttr);

    if (auto config = getLoweringConfig(mmaOp)) {
      setLoweringConfig(concreteMmaOp, config);
    }

    if (fragment != MMAFragment::Acc) {
      rewriter.replaceOp(mmaOp, concreteMmaOp);
      return success();
    }

    // For the Acc operand, the result needs to be collapsed back to the
    // original type so that types match with consumers.
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        mmaOp, mmaOp.getOutputs().front().getType(),
        concreteMmaOp.getResults().front(), reassociations);

    return success();
  }

private:
  MMAFragment fragment;
};

} // namespace

void ConcretizeMmaShapesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet patterns(context);
  if (concretizeInputs) {
    patterns.insert<ConcretizeMmaOperandShape>(context, MMAFragment::Lhs);
    patterns.insert<ConcretizeMmaOperandShape>(context, MMAFragment::Rhs);
  }
  if (concretizeResult) {
    patterns.insert<ConcretizeMmaOperandShape>(context, MMAFragment::Acc);
  }
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
