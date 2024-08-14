// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
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
} // namespace

struct ConcretizeMmaOperandShape final : OpRewritePattern<MultiMmaOp> {
  using OpRewritePattern::OpRewritePattern;

  ConcretizeMmaOperandShape(MLIRContext *context, MMAFragment fragment)
      : OpRewritePattern<MultiMmaOp>(context), fragment(fragment) {}

  LogicalResult matchAndRewrite(MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    if (!mmaOp.hasTensorSemantics()) {
      return failure();
    }

    // Get the right operand and permutation for the `fragment`.
    Value operand;
    std::optional<ArrayRef<int64_t>> permutation;
    switch (fragment) {
    case MMAFragment::Lhs:
      operand = mmaOp.getLhs();
      permutation = mmaOp.getLhsPermutation();
      break;
    case MMAFragment::Rhs:
      operand = mmaOp.getRhs();
      permutation = mmaOp.getRhsPermutation();
      break;
    case MMAFragment::Acc:
      operand = mmaOp.getAcc();
      permutation = mmaOp.getAccPermutation();
      break;
    }

    // Get the reassociation indices and result type of the expand_shape op.
    MmaInterfaceAttr kind = mmaOp.getKind();
    SmallVector<ReassociationIndices> reassociations;
    RankedTensorType concreteType;
    if (failed(kind.materializeOperandConcreteShape(rewriter, fragment, operand,
                                                    permutation, reassociations,
                                                    concreteType))) {
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
    // multi_mma operand.
    auto expandPerm =
        [&](std::optional<ArrayRef<int64_t>> perm, MMAFragment frag,
            int64_t outerRank) -> std::optional<DenseI64ArrayAttr> {
      if (!perm.has_value()) {
        return std::nullopt;
      }
      if (frag != fragment) {
        return rewriter.getDenseI64ArrayAttr(perm.value());
      }
      SmallVector<ReassociationIndices> innerReInds(
          reassociations.begin() + outerRank, reassociations.end());
      for (auto &reInd : innerReInds) {
        for (auto &idx : reInd) {
          idx -= outerRank;
        }
      }
      SmallVector<int64_t> expandedPerm;
      for (auto reInd : applyPermutation(innerReInds, perm.value())) {
        expandedPerm.append(reInd);
      }
      return rewriter.getDenseI64ArrayAttr(expandedPerm);
    };
    std::optional<DenseI64ArrayAttr> lhsPerm = expandPerm(
        mmaOp.getLhsPermutation(), MMAFragment::Lhs, mmaOp.getLhsOuterRank());
    std::optional<DenseI64ArrayAttr> rhsPerm = expandPerm(
        mmaOp.getRhsPermutation(), MMAFragment::Rhs, mmaOp.getRhsOuterRank());
    std::optional<DenseI64ArrayAttr> accPerm = expandPerm(
        mmaOp.getAccPermutation(), MMAFragment::Acc, mmaOp.getAccOuterRank());

    // Create the new multi_mma op with the concrete type.
    auto concreteMmaOp = rewriter.create<MultiMmaOp>(
        loc,
        /*lhs=*/fragment == MMAFragment::Lhs ? concreteOperand : mmaOp.getLhs(),
        /*rhs=*/fragment == MMAFragment::Rhs ? concreteOperand : mmaOp.getRhs(),
        /*acc=*/fragment == MMAFragment::Acc ? concreteOperand : mmaOp.getAcc(),
        mmaOp.getIndexingMaps(), mmaOp.getIteratorTypes(), mmaOp.getKind(),
        lhsPerm, rhsPerm, accPerm);

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
        mmaOp, mmaOp.getAccType(), concreteMmaOp.getResult(), reassociations);

    return success();
  }

private:
  MMAFragment fragment;
};

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
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
