// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops,
// taking care of CHLO's broadcasting semantics

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo-iree/Conversion/Rewriters.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCUSTOMCALLS
#include "stablehlo-iree/Conversion/Passes.h.inc"

namespace {

// Computes householder using vector `v` and a `tau` matrice
// with the k-th element. See householder transformation as below:
// https://en.wikipedia.org/wiki/Householder_transformation
static Value computeHouseholder(Value v, Value tau, Value k,
                                ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());

  SmallVector<int64_t> hShape(vTy.getShape());
  hShape.push_back(hShape.back());

  auto hTy = RankedTensorType::get(hShape, vTy.getElementType());
  Value empty = b.create<tensor::EmptyOp>(hShape, vTy.getElementType());

  auto outMap = b.getMultiDimIdentityMap(hShape.size());

  SmallVector<AffineExpr> exprs;
  for (int i = 0, s = vTy.getRank(); i < s; ++i) {
    exprs.push_back(b.getAffineDimExpr(i));
  }

  AffineMap vMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  exprs.back() = b.getAffineDimExpr(vTy.getRank());
  AffineMap vTMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  SmallVector<AffineMap> affineMaps = {vMap, vTMap, outMap};

  SmallVector<utils::IteratorType> iterTypes(hShape.size(),
                                             utils::IteratorType::parallel);

  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(vTy.getElementType()));
  Value one =
      b.create<arith::ConstantOp>(b.getFloatAttr(vTy.getElementType(), 1.0));

  return b
      .create<linalg::GenericOp>(
          hTy, ValueRange{v, v}, empty, affineMaps, iterTypes,
          [&](OpBuilder &bb, Location loc, ValueRange args) {
            ImplicitLocOpBuilder b(loc, bb);
            SmallVector<Value> indices;
            for (int i = 0, s = hTy.getRank(); i < s; ++i) {
              indices.push_back(b.create<linalg::IndexOp>(loc, i));
            }

            SmallVector<Value> tauIndices(indices.begin(), indices.end() - 2);
            tauIndices.push_back(k);
            Value t = b.create<tensor::ExtractOp>(tau, tauIndices);

            // Generates the lower triangularization of the matrix with
            // one values on the diagonal.
            auto tri = [&](Value v, Value i) {
              Value eq =
                  b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, i, k);
              Value lt =
                  b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, i, k);
              Value sel = b.create<arith::SelectOp>(eq, one, v);
              return b.create<arith::SelectOp>(lt, zero, sel);
            };

            Value v = tri(args[0], indices[indices.size() - 2]);
            Value vT = tri(args[1], indices[indices.size() - 1]);

            Value h = b.create<arith::MulFOp>(v, vT);
            h = b.create<arith::MulFOp>(h, t);

            Value isDiag = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                   indices[indices.size() - 2],
                                                   indices[indices.size() - 1]);
            Value diag = b.create<arith::SelectOp>(isDiag, one, zero);
            Value sub = b.create<arith::SubFOp>(diag, h);

            b.create<linalg::YieldOp>(sub);
          })
      .getResult(0);
}

// Slices the k-th column of matrix and computes the householder transformation
// for using the `tau` value.
static Value computeHouseholderSlice(Value matrix, Value tau, Value k,
                                     ImplicitLocOpBuilder b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  int rank = matrixTy.getRank();

  SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
  SmallVector<int64_t> vShape(matrixTy.getShape());
  vShape[vShape.size() - 1] = 1;

  SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  vOffsets[vOffsets.size() - 1] = k;

  SmallVector<OpFoldResult> vSizes;
  for (auto v : vShape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
  Value v = b.create<tensor::ExtractSliceOp>(sliceTy, matrix, vOffsets, vSizes,
                                             vStrides);

  SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < rank - 2; ++i) {
    reass.push_back({i});
  }
  reass.push_back({rank - 2, rank - 1});

  ArrayRef<int64_t> collapseVShape(vShape.begin(), vShape.end() - 1);
  auto collapseVTy =
      RankedTensorType::get(collapseVShape, matrixTy.getElementType());
  Value collapseV = b.create<tensor::CollapseShapeOp>(collapseVTy, v, reass);

  Value householder = computeHouseholder(collapseV, tau, k, b);
  return householder;
}

struct HouseholderReflectorRewriter final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern<mlir::stablehlo::CustomCallOp>::OpRewritePattern;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "ProductOfElementaryHouseholderReflectors") {
      return rewriter.notifyMatchFailure(
          op, "not ProductOfElementaryHouseholderReflectors");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto matrix = op.getOperand(0);
    auto tau = op.getOperand(1);
    auto matrixTy = cast<ShapedType>(matrix.getType());
    auto tauTy = cast<ShapedType>(tau.getType());
    auto rank = matrixTy.getRank();

    if (isa<ComplexType>(matrixTy.getElementType())) {
      return rewriter.notifyMatchFailure(op, "complex types not supported");
    }

    if (rank < 2) {
      return rewriter.notifyMatchFailure(op, "requires minimum rank 2 matrix");
    }

    // Implementation needs to be checked to work with variable dimension
    // lengths. Should be relatively straightforward.
    if (!matrixTy.hasStaticShape() || !tauTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "not supported for dynamic shapes");
    }

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);
    Value k = b.create<arith::ConstantIndexOp>(tauTy.getShape().back());
    Value householder0 = computeHouseholderSlice(matrix, tau, zero, b);
    auto scf = b.create<scf::ForOp>(
        one, k, one, ValueRange{householder0},
        [&](OpBuilder &bb, Location loc, Value iv, ValueRange args) {
          ImplicitLocOpBuilder b(loc, bb);
          Value householder = computeHouseholderSlice(matrix, tau, iv, b);

          std::vector<int64_t> batch(rank - 2);
          for (int i = 0; i < rank - 2; ++i)
            batch[i] = i;
          std::vector<int64_t> lhsContract = {rank - 1};
          std::vector<int64_t> rhsContract = {rank - 2};

          auto dotNums = mlir::stablehlo::DotDimensionNumbersAttr::get(
              b.getContext(), batch, batch, lhsContract, rhsContract);
          Value dot = b.create<mlir::stablehlo::DotGeneralOp>(
              householder0.getType(), args[0], householder, dotNums, nullptr);
          b.create<scf::YieldOp>(loc, dot);
        });

    SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
    SmallVector<int64_t> vShape(matrixTy.getShape());
    SmallVector<OpFoldResult> vSizes;
    for (auto v : vShape) {
      vSizes.push_back(b.getIndexAttr(v));
    }

    auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
    Value v = b.create<tensor::ExtractSliceOp>(sliceTy, scf.getResult(0),
                                               vOffsets, vSizes, vStrides);

    rewriter.replaceOp(op, v);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition.
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOCustomCalls final
    : impl::LegalizeStableHLOCustomCallsBase<LegalizeStableHLOCustomCalls> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    mlir::stablehlo::StablehloDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<HouseholderReflectorRewriter>(ctx);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
