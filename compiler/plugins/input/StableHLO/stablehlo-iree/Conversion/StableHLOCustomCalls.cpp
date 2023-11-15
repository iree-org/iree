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
#include "mlir/Dialect/Math/IR/Math.h"
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

static Value makeDot(Value lhs, Value rhs, int64_t nbatch, int64_t ncontract,
                     ImplicitLocOpBuilder b) {
  auto lhsTy = cast<ShapedType>(lhs.getType());
  auto rhsTy = cast<ShapedType>(rhs.getType());

  std::vector<int64_t> batch;
  std::vector<int64_t> outShape;
  for (int i = 0; i < nbatch; ++i) {
    batch.push_back(i);
    outShape.push_back(lhsTy.getDimSize(i));
  }

  for (int i = nbatch; i < lhsTy.getRank() - ncontract; ++i) {
    outShape.push_back(lhsTy.getDimSize(i));
  }

  for (int i = nbatch; i < rhsTy.getRank() - ncontract; ++i) {
    outShape.push_back(rhsTy.getDimSize(i));
  }

  std::vector<int64_t> lhsContract;
  for (int i = lhsTy.getRank() - ncontract; i < lhsTy.getRank(); ++i) {
    lhsContract.push_back(i);
  }

  std::vector<int64_t> rhsContract;
  for (int i = rhsTy.getRank() - ncontract; i < rhsTy.getRank(); ++i) {
    rhsContract.push_back(i);
  }

  auto dotNums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      b.getContext(), batch, batch, lhsContract, rhsContract);
  return b.create<mlir::stablehlo::DotGeneralOp>(
      RankedTensorType::get(outShape, lhsTy.getElementType()), lhs, rhs,
      dotNums, nullptr);
}

static Value collapseBackDims(Value v, int64_t ndims, ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());
  int64_t rank = vTy.getRank();
  SmallVector<int64_t> dims;
  dims.reserve(rank);

  llvm::SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < rank - ndims; ++i) {
    dims.push_back(vTy.getDimSize(i));
    reass.push_back({i});
  }

  for (int64_t i = rank - ndims; i < rank; i++) {
    dims.back() *= vTy.getDimSize(i);
    if (!reass.empty())
      reass.back().push_back(i);
  }

  auto newTy = RankedTensorType::get(dims, vTy.getElementType());
  return b.create<tensor::CollapseShapeOp>(newTy, v, reass);
}
static Value expandDim(Value v, int64_t dim, ArrayRef<int64_t> sizes,
                       ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());
  int64_t rank = vTy.getRank();
  SmallVector<int64_t> dims;
  dims.reserve(rank + sizes.size());

  llvm::SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < dim; ++i) {
    reass.push_back({i});
    dims.push_back(vTy.getDimSize(i));
  }

  reass.push_back({});
  for (auto dim : sizes) {
    reass.back().push_back(dims.size());
    dims.push_back(dim);
  }

  for (int i = dim + 1; i < rank; ++i) {
    reass.push_back({static_cast<int64_t>(dims.size())});
    dims.push_back(vTy.getDimSize(i));
  }

  auto newTy = RankedTensorType::get(dims, vTy.getElementType());
  return b.create<tensor::ExpandShapeOp>(newTy, v, reass);
}

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

llvm::SmallVector<OpFoldResult> makeIndexArray(llvm::SmallVector<int64_t> idx,
                                               ImplicitLocOpBuilder b) {
  llvm::SmallVector<OpFoldResult> res;
  res.reserve(idx.size());
  for (auto i : idx) {
    res.push_back(b.getIndexAttr(i));
  }

  return res;
}

Value extractSlice(Value v, llvm::ArrayRef<OpFoldResult> offsets,
                   llvm::ArrayRef<int64_t> sizes, llvm::ArrayRef<int64_t> dims,
                   ImplicitLocOpBuilder b) {
  assert(offsets.size() == dims.size());
  assert(sizes.size() == dims.size());

  auto vTy = cast<ShapedType>(v.getType());
  auto rank = vTy.getRank();

  llvm::SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  llvm::SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));

  llvm::SmallVector<int64_t> shape(vTy.getShape());
  for (int i = 0; i < dims.size(); ++i) {
    shape[dims[i]] = sizes[i];
    vOffsets[dims[i]] = offsets[i];
  }

  llvm::SmallVector<OpFoldResult> vSizes;
  for (auto v : shape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  // Slice the top-left most element:
  auto sliceTy = RankedTensorType::get(shape, vTy.getElementType());
  return b.create<tensor::ExtractSliceOp>(sliceTy, v, vOffsets, vSizes,
                                          vStrides);
}

Value insertSlice(Value dest, Value update,
                  llvm::ArrayRef<OpFoldResult> offsets,
                  llvm::ArrayRef<int64_t> dims, ImplicitLocOpBuilder b) {
  assert(offsets.size() == dims.size());

  auto destTy = cast<ShapedType>(dest.getType());
  auto updateTy = cast<ShapedType>(update.getType());
  auto rank = destTy.getRank();

  llvm::SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  llvm::SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));

  llvm::SmallVector<int64_t> shape(updateTy.getShape());
  for (int i = 0; i < dims.size(); ++i) {
    vOffsets[dims[i]] = offsets[i];
  }

  llvm::SmallVector<OpFoldResult> vSizes;
  for (auto v : shape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  return b.create<tensor::InsertSliceOp>(update, dest, vOffsets, vSizes,
                                         vStrides);
}

AffineMap getAffineMap(llvm::ArrayRef<int64_t> skipDims, int64_t rank,
                       ImplicitLocOpBuilder b) {
  llvm::SmallVector<AffineExpr> exprs;
  int skipIdx = 0;
  for (int i = 0; i < rank; ++i) {
    if (skipIdx < skipDims.size() && i == skipDims[skipIdx]) {
      ++skipIdx;
      continue;
    }
    exprs.push_back(b.getAffineDimExpr(i));
  }

  return AffineMap::get(rank, 0, exprs, b.getContext());
}

// Rewriter for a cholesky decomposition using linalg operations.
// It involves column-by-column updating the cholesky decomposition
// and transposing the result if we require an upper triagle matrix.
//
// This solution is equivalent to the python code below:
//   def cholesky(A):
//     Rank = A.rank
//     N = A.shape[Rank - 1]
//     L = np.zeros_like(A)
//     for i in range(N):
//       L[i,i] = sqrt(A[i,i] - dot_product(L[i,:], L[i,:]))
//       L[:,i] = (A[:,i] - matmul(conjg(L[i,:]), L)) / L[i,i]
struct CholeskyRewriter final : OpRewritePattern<mlir::stablehlo::CholeskyOp> {
  using OpRewritePattern<mlir::stablehlo::CholeskyOp>::OpRewritePattern;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CholeskyOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto operand = op.getOperand();
    auto operandTy = cast<ShapedType>(operand.getType());
    auto resultTy = cast<ShapedType>(op.getType());
    const auto rank = operandTy.getRank();
    if (!operandTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "not supported for dynamic shapes");
    }

    Value left =
        extractSlice(operand, makeIndexArray({0}, b), {1}, {rank - 1}, b);
    Value topleft = extractSlice(operand, makeIndexArray({0, 0}, b), {1, 1},
                                 {rank - 1, rank - 2}, b);
    topleft = collapseBackDims(topleft, 2, b);

    AffineMap topleftMap = getAffineMap({rank - 2, rank - 1}, rank, b);
    AffineMap leftMap = b.getMultiDimIdentityMap(rank);
    AffineMap outMap = b.getMultiDimIdentityMap(rank);

    SmallVector<AffineMap> affineMaps = {topleftMap, leftMap, outMap};
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);
    Value emptyCol = b.create<tensor::EmptyOp>(
        cast<ShapedType>(left.getType()).getShape(), resultTy.getElementType());

    Value generic = b.create<linalg::GenericOp>(
                         emptyCol.getType(), ValueRange{topleft, left},
                         emptyCol, affineMaps, iterTypes,
                         [&](OpBuilder &bb, Location loc, ValueRange args) {
                           Value sqrt = bb.create<math::SqrtOp>(loc, args[0]);
                           Value div =
                               bb.create<arith::DivFOp>(loc, args[1], sqrt);
                           return bb.create<linalg::YieldOp>(loc, div);
                         })
                        .getResult(0);

    // We initialize the tensor with all zeros.
    Value zeroF =
        b.create<arith::ConstantOp>(b.getZeroAttr(resultTy.getElementType()));
    Value empty = b.create<tensor::SplatOp>(resultTy, zeroF);

    Value insert = insertSlice(empty, generic, {}, {}, b);

    // Iterate row-by-row in the input matrix and upate the estimated
    // cholesky decomposition.
    Value one = b.create<arith::ConstantIndexOp>(1);
    Value N = b.create<arith::ConstantIndexOp>(operandTy.getShape().back());
    Value scf =
        b.create<scf::ForOp>(
             one, N, one, ValueRange{insert},
             [&](OpBuilder &bb, Location loc, Value col, ValueRange args) {
               ImplicitLocOpBuilder b(loc, bb);
               // vecvec = dot(L[i, :], L[i, :]):
               Value outSlice =
                   extractSlice(args[0], {col}, {1}, {rank - 2}, b);
               outSlice = collapseBackDims(outSlice, 1, b);
               Value vecvec = makeDot(outSlice, outSlice, rank - 2, 1, b);

               // matvec = dot(L[:, :], L[i, :]):
               Value inSlice = extractSlice(operand, {col}, {1}, {rank - 2}, b);
               inSlice = collapseBackDims(inSlice, 1, b);
               Value matvec = makeDot(args[0], outSlice, rank - 2, 1, b);

               // diag = A[i, i]
               Value diag = extractSlice(operand, {col, col}, {1, 1},
                                         {rank - 2, rank - 1}, b);
               diag = collapseBackDims(diag, 2, b);

               AffineMap scalarMap = getAffineMap({rank - 2}, rank - 1, b);
               AffineMap outMap = b.getMultiDimIdentityMap(rank - 1);
               SmallVector<AffineMap> affineMaps = {
                   outMap, outMap, outMap, scalarMap, scalarMap, outMap};
               SmallVector<utils::IteratorType> iterTypes(
                   rank - 1, utils::IteratorType::parallel);

               ShapedType colTy = RankedTensorType::get(
                   cast<ShapedType>(resultTy).getShape().drop_back(),
                   resultTy.getElementType());
               Value empty = b.create<tensor::EmptyOp>(colTy.getShape(),
                                                       colTy.getElementType());

               // Compute the new column values:
               //   C_i = sqrt(diag - vecvec)
               //   C[j] = j <= i ? matvec[j] - L[i, j] : 0
               Value generic =
                   b.create<linalg::GenericOp>(
                        colTy,
                        ValueRange{outSlice, matvec, inSlice, diag, vecvec},
                        empty, affineMaps, iterTypes,
                        [&](OpBuilder &bb, Location loc, ValueRange args) {
                          Value idx = b.create<linalg::IndexOp>(rank - 2);
                          Value cmp = b.create<arith::CmpIOp>(
                              arith::CmpIPredicate::uge, idx, col);
                          Value subSlice =
                              b.create<arith::SubFOp>(args[2], args[1]);
                          Value subDiag =
                              b.create<arith::SubFOp>(args[3], args[4]);
                          Value sqrt = b.create<math::SqrtOp>(subDiag);
                          Value div = b.create<arith::DivFOp>(subSlice, sqrt);
                          Value sel =
                              b.create<arith::SelectOp>(cmp, div, zeroF);
                          b.create<linalg::YieldOp>(sel);
                        })
                       .getResult(0);

               // Update the output matrix:
               // L[j, i] = C[j]
               generic = expandDim(generic, rank - 2,
                                   {colTy.getShape().back(), 1}, b);
               Value insert =
                   insertSlice(args[0], generic, {col}, {rank - 1}, b);
               return b.create<scf::YieldOp>(insert);
             })
            .getResult(0);

    // Transpose the final two dimensions if we require an upper triangular
    // matrix.
    if (!op.getLower()) {
      llvm::SmallVector<int64_t> shape(operandTy.getShape());
      llvm::SmallVector<int64_t> perms(operandTy.getRank());

      for (int i = 0; i < operandTy.getRank(); i++) {
        perms[i] = i;
      }

      std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
      std::swap(perms[perms.size() - 2], perms[perms.size() - 1]);
      scf = b.create<mlir::stablehlo::TransposeOp>(
          RankedTensorType::get(shape, operandTy.getElementType()), scf,
          b.getI64TensorAttr(perms));
    }

    rewriter.replaceOp(op, scf);
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
    patterns.add<CholeskyRewriter, HouseholderReflectorRewriter>(ctx);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
