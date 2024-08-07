// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_CONCRETIZEMMASHAPESPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct ConcretizeMmaShapesPass final
    : impl::ConcretizeMmaShapesPassBase<ConcretizeMmaShapesPass> {
  void runOnOperation() override;
};
} // namespace

struct ConcretizeMmaShapes final : OpRewritePattern<IREE::GPU::MultiMmaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    IREE::GPU::MmaInterfaceAttr kind = mmaOp.getKind();
    SmallVector<ReassociationIndices> lhsReassociations, rhsReassociations,
        accReassociations;
    RankedTensorType lhsConcreteType, rhsConcreteType, accConcreteType;
    if (failed(kind.materializeOperandConcreteShape(
            rewriter, IREE::GPU::MMAFragment::Lhs, mmaOp.getLhs(),
            mmaOp.getLhsPermutation(), lhsReassociations, lhsConcreteType))) {
      return failure();
    }
    if (failed(kind.materializeOperandConcreteShape(
            rewriter, IREE::GPU::MMAFragment::Rhs, mmaOp.getRhs(),
            mmaOp.getRhsPermutation(), rhsReassociations, rhsConcreteType))) {
      return failure();
    }
    if (failed(kind.materializeOperandConcreteShape(
            rewriter, IREE::GPU::MMAFragment::Acc, mmaOp.getAcc(),
            mmaOp.getAccPermutation(), accReassociations, accConcreteType))) {
      return failure();
    }
    Location loc = mmaOp->getLoc();
    Value concreteLhs =
        rewriter
            .create<tensor::ExpandShapeOp>(loc, lhsConcreteType, mmaOp.getLhs(),
                                           lhsReassociations)
            .getResult();
    Value concreteRhs =
        rewriter
            .create<tensor::ExpandShapeOp>(loc, rhsConcreteType, mmaOp.getRhs(),
                                           rhsReassociations)
            .getResult();
    Value concreteAcc =
        rewriter
            .create<tensor::ExpandShapeOp>(loc, accConcreteType, mmaOp.getAcc(),
                                           accReassociations)
            .getResult();

    std::optional<DenseI64ArrayAttr> lhsPerm, rhsPerm, accPerm;
    if (mmaOp.getLhsPermutation().has_value())
      lhsPerm = mmaOp.getLhsPermutationAttr();
    if (mmaOp.getRhsPermutation().has_value())
      rhsPerm = mmaOp.getRhsPermutationAttr();
    if (mmaOp.getAccPermutation().has_value())
      accPerm = mmaOp.getAccPermutationAttr();

    auto concreteMmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
        loc, concreteLhs, concreteRhs, concreteAcc, mmaOp.getIndexingMaps(),
        mmaOp.getIteratorTypes(), mmaOp.getKind(), lhsPerm, rhsPerm, accPerm);

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        mmaOp, mmaOp.getAccType(), concreteMmaOp.getResult(),
        accReassociations);

    return success();
  }
};

void ConcretizeMmaShapesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  RewritePatternSet patterns(context);
  patterns.insert<ConcretizeMmaShapes>(context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
