// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering the StableHLO general dot op to the dot op.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_DOTGENERALTODOT
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {
Value transposeReshape(Value arg, Location loc,
                       llvm::ArrayRef<int64_t> leftDims,
                       llvm::ArrayRef<int64_t> rightDims,
                       llvm::ArrayRef<int64_t> argShape,
                       PatternRewriter &rewriter) {
  Type elementType = getElementTypeOrSelf(arg.getType());

  int64_t leftSize = 1;
  for (int64_t dim : leftDims) {
    leftSize = (ShapedType::isDynamic(argShape[dim]) || leftSize < 0)
                   ? ShapedType::kDynamic
                   : leftSize * argShape[dim];
  }

  int64_t rightSize = 1;
  for (int64_t dim : rightDims) {
    rightSize = (ShapedType::isDynamic(argShape[dim]) || rightSize < 0)
                    ? ShapedType::kDynamic
                    : rightSize * argShape[dim];
  }

  // Generate the transpose permutation attribute.
  auto transposePermutation =
      llvm::to_vector<5>(llvm::concat<const int64_t>(leftDims, rightDims));

  TensorType transposePermutationType =
      RankedTensorType::get({static_cast<int64_t>(transposePermutation.size())},
                            rewriter.getIntegerType(64));

  auto transposePermutationAttr =
      llvm::cast<DenseIntElementsAttr>(DenseIntElementsAttr::get(
          transposePermutationType, llvm::ArrayRef(transposePermutation)));

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 5> transposedShape;
  for (int64_t val : transposePermutation) {
    transposedShape.push_back(argShape[val]);
  }

  // If there are only a single pair of contracting dimensions and the output
  // rank is two we can skip a needless reshape.
  bool noReshape = transposedShape.size() == 2 && leftDims.size() == 1 &&
                   rightDims.size() == 1;

  // Construct transpose. If no reshape is needed, we are done.
  auto transposeType = RankedTensorType::get(transposedShape, elementType);
  Value transposeResult = rewriter.create<mlir::stablehlo::TransposeOp>(
      loc, transposeType, arg, transposePermutationAttr);
  if (noReshape)
    return transposeResult;

  // Return the final result.
  auto reshapedType = RankedTensorType::get({leftSize, rightSize}, elementType);

  if (reshapedType.hasStaticShape()) {
    return rewriter.create<mlir::stablehlo::ReshapeOp>(loc, reshapedType,
                                                       transposeResult);
  }

  SmallVector<Value> reshapeDims;
  auto multiplyDynamicDims = [&](llvm::ArrayRef<int64_t> dims) -> Value {
    Value dynamicSize = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
        loc, arg, rewriter.getI64IntegerAttr(dims.front()));
    Value dynamicSizeReshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynamicSize);
    for (auto idx : dims.drop_front()) {
      Value dim = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
          loc, arg, rewriter.getI64IntegerAttr(idx));
      Value dimReshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({1}, rewriter.getI32Type()), dim);
      dynamicSizeReshaped = rewriter.create<mlir::stablehlo::MulOp>(
          loc, dynamicSizeReshaped, dimReshaped);
    }
    return dynamicSizeReshaped;
  };

  if (leftSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(leftDims));
  } else {
    reshapeDims.push_back(rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, rewriter.getI32TensorAttr(leftSize)));
  }

  if (rightSize < 0) {
    reshapeDims.push_back(multiplyDynamicDims(rightDims));
  } else {
    reshapeDims.push_back(rewriter.create<mlir::stablehlo::ConstantOp>(
        loc, rewriter.getI32TensorAttr(rightSize)));
  }

  Value reshapeDimsTensor = rewriter.create<mlir::stablehlo::ConcatenateOp>(
      loc, RankedTensorType::get({2}, rewriter.getI32Type()), reshapeDims,
      rewriter.getI64IntegerAttr(0));
  return rewriter.create<mlir::stablehlo::DynamicReshapeOp>(
      loc, reshapedType, transposeResult, reshapeDimsTensor);
}

Value processDotArg(Value arg, Location loc, ArrayRef<int64_t> contractDimsAttr,
                    bool outerDimsFirst, PatternRewriter &rewriter) {
  auto shape = llvm::cast<ShapedType>(arg.getType()).getShape();

  llvm::SmallVector<bool, 5> isOuterDim;
  isOuterDim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contractDims;
  for (auto dim : contractDimsAttr) {
    contractDims.push_back(dim);
    isOuterDim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outerDims;
  for (const auto &it : llvm::enumerate(isOuterDim)) {
    if (it.value()) {
      outerDims.push_back(it.index());
    }
  }

  if (outerDimsFirst) {
    return transposeReshape(arg, loc, outerDims, contractDims, shape, rewriter);
  }

  return transposeReshape(arg, loc, contractDims, outerDims, shape, rewriter);
}

struct GeneralDotRemoveBatch final
    : OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsTy = cast<ShapedType>(op.getLhs().getType());
    auto rhsTy = cast<ShapedType>(op.getRhs().getType());
    auto ty = cast<ShapedType>(op.getType());

    if (!ty.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "does not have static shape");
    }

    auto dimNumbers = op.getDotDimensionNumbers();
    if (dimNumbers.getLhsBatchingDimensions().size() != 1 ||
        dimNumbers.getLhsBatchingDimensions().size() != 1) {
      return rewriter.notifyMatchFailure(op, "non-unary batch dimension");
    }

    if (dimNumbers.getLhsBatchingDimensions().front() != 0 ||
        dimNumbers.getRhsBatchingDimensions().front() != 0) {
      return rewriter.notifyMatchFailure(op, "not first dim on lhs/rhs");
    }

    if (lhsTy.getDimSize(0) != 1 || rhsTy.getDimSize(0) != 1) {
      return rewriter.notifyMatchFailure(op, "not unary batch size");
    }

    // We no longer include the batch dimension of 1.
    llvm::SmallVector<int64_t> newLhsContractingDims;
    for (auto dim : dimNumbers.getLhsContractingDimensions())
      newLhsContractingDims.push_back(dim - 1);

    llvm::SmallVector<int64_t> newRhsContractingDims;
    for (auto dim : dimNumbers.getRhsContractingDimensions())
      newRhsContractingDims.push_back(dim - 1);

    auto lhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        op.getLoc(), lhsTy.clone(lhsTy.getShape().drop_front()), op.getLhs());

    auto rhs = rewriter.create<mlir::stablehlo::ReshapeOp>(
        op.getLoc(), rhsTy.clone(rhsTy.getShape().drop_front()), op.getRhs());

    auto newDimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/
        newLhsContractingDims,
        /*rhsContractingDimensions=*/
        newRhsContractingDims);

    auto dot = rewriter.create<mlir::stablehlo::DotGeneralOp>(
        op.getLoc(), ty.clone(ty.getShape().drop_front()), lhs, rhs,
        newDimNumbers, op.getPrecisionConfigAttr());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, ty,
                                                            dot.getResult());
    return success();
  }
};

struct GeneralDotConvert final
    : OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.
  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dotNumbers = op.getDotDimensionNumbers();
    if (!dotNumbers.getLhsBatchingDimensions().empty() ||
        !dotNumbers.getRhsBatchingDimensions().empty()) {
      return failure();
    }

    ArrayAttr precisionConfig;
    auto opPrecisionConfig = op.getPrecisionConfig();
    if (opPrecisionConfig.has_value())
      precisionConfig = *opPrecisionConfig;

    auto resultTy = cast<ShapedType>(op.getType());

    ArrayRef<int64_t> lhsContractingDims =
        dotNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotNumbers.getRhsContractingDimensions();

    TypedValue<TensorType> lhs = op.getLhs();
    TypedValue<TensorType> rhs = op.getRhs();

    RankedTensorType lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy)
      return failure();

    // The StableHLO dot operator directly supports a vector dot product
    // (two vectors reduce into a scalar) as well as a matrix vector
    // product (a matrix and vector reduce into a vector) without any
    // need for reshaping. We handle those special cases first, before
    // entering the general logic that reduces into a matrix.
    if (lhsTy.hasStaticShape() && rhsTy.hasStaticShape() &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
      if (lhsTy.getRank() == 1 && rhsTy.getRank() == 1) {
        // Vector-vector, reduces into scalar.
        assert(lhsContractingDims[0] == 0 && rhsContractingDims[0] == 0);
        ShapedType newTy = RankedTensorType::get({}, resultTy.getElementType());
        rewriter.replaceOpWithNewOp<mlir::stablehlo::DotOp>(op, newTy, lhs, rhs,
                                                            precisionConfig);
        return success();
      }
      if (lhsTy.getRank() == 2 && rhsTy.getRank() == 1 &&
          lhsContractingDims[0] == 1) {
        // Matrix-vector, reduces into vector.
        assert(rhsContractingDims[0] == 0);
        ShapedType newTy = RankedTensorType::get({lhsTy.getShape()[0]},
                                                 resultTy.getElementType());
        rewriter.replaceOpWithNewOp<mlir::stablehlo::DotOp>(op, newTy, lhs, rhs,
                                                            precisionConfig);
        return success();
      }
      if (lhsTy.getRank() == 2 && rhsTy.getRank() == 2 &&
          lhsContractingDims[0] == 1 && rhsContractingDims[0] == 0) {
        // Matrix-matrix, reduces into matrix. Note that for dense cases, this
        // rewriting rule simply provides a shortcut for what is to follow
        // (modulo optimizing the trivial transpose/reshape operations). For
        // sparse cases, however, this rewriting preserves the output sparsity
        // that was explicitly given for the general dot operation.
        Value newDotOp = rewriter.create<mlir::stablehlo::DotOp>(
            loc, resultTy, lhs, rhs, precisionConfig);
        if (auto enc = sparse_tensor::getSparseTensorEncoding(resultTy)) {
          newDotOp.setType(RankedTensorType::get(
              resultTy.getShape(), resultTy.getElementType(), enc));
        }
        rewriter.replaceOp(op, newDotOp);
        return success();
      }
    }

    // For any sparse situation, don't use any of the following rules, since
    // transposing and reshaping is not without cost. Instead, rely on the
    // default linalg lowering that follows later in the pipeline.
    if (sparse_tensor::hasAnySparseOperandOrResult(op))
      return failure();

    // Compute the, possibly, transposed-reshaped operands.
    lhs = cast<mlir::TypedValue<mlir::TensorType>>(processDotArg(
        lhs, loc, lhsContractingDims, /*outerDimsFirst=*/true, rewriter));
    rhs = cast<mlir::TypedValue<mlir::TensorType>>(processDotArg(
        rhs, loc, rhsContractingDims, /*outerDimsFirst=*/false, rewriter));

    // Accept only static shaped types.
    auto lhsShapeType = dyn_cast_or_null<ShapedType>(lhs.getType());
    auto rhsShapeType = dyn_cast_or_null<ShapedType>(rhs.getType());
    if (!lhsShapeType || !rhsShapeType)
      return failure();

    // Generate new dot operator on expanded types.
    ShapedType newTy = RankedTensorType::get(
        {lhsShapeType.getShape()[0], rhsShapeType.getShape()[1]},
        resultTy.getElementType());
    Value newDotOp = rewriter.create<mlir::stablehlo::DotOp>(
        loc, newTy, lhs, rhs, precisionConfig);
    if (static_cast<int64_t>(lhsContractingDims.size()) ==
            lhsTy.getRank() - 1 &&
        static_cast<int64_t>(rhsContractingDims.size()) ==
            rhsTy.getRank() - 1) {
      rewriter.replaceOp(op, newDotOp);
      return success();
    }

    // We can avoid all the computation below if we know the static shape.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, resultTy,
                                                              newDotOp);
      return success();
    }

    llvm::SmallVector<int64_t> staticDims;
    llvm::SmallVector<Value> dynDims;

    auto getDynamicDims = [&](Value arg,
                              llvm::ArrayRef<int64_t> contractingDims) {
      RankedTensorType ty = llvm::cast<RankedTensorType>(arg.getType());
      int index = 0;
      for (int64_t contractingDim : contractingDims) {
        for (; index < contractingDim; ++index) {
          staticDims.push_back(ty.getDimSize(index));
          Value dynDim = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
              loc, arg, rewriter.getI64IntegerAttr(index));
          Value dynDimReshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
              loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynDim);
          dynDims.push_back(dynDimReshaped);
        }
        index++;
      }

      for (; index < ty.getRank(); ++index) {
        staticDims.push_back(ty.getDimSize(index));
        Value dynDim = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
            loc, arg, rewriter.getI64IntegerAttr(index));
        Value dynDimReshaped = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc, RankedTensorType::get({1}, rewriter.getI32Type()), dynDim);
        dynDims.push_back(dynDimReshaped);
      }
    };

    getDynamicDims(op.getLhs(), lhsContractingDims);
    getDynamicDims(op.getRhs(), rhsContractingDims);

    Value reshapeDimsTensor = rewriter.create<mlir::stablehlo::ConcatenateOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(dynDims.size())},
                              rewriter.getI32Type()),
        dynDims, rewriter.getI64IntegerAttr(0));

    Value result = rewriter.create<mlir::stablehlo::DynamicReshapeOp>(
        loc, RankedTensorType::get(staticDims, resultTy.getElementType()),
        newDotOp, reshapeDimsTensor);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DotVectorOptimization final : OpRewritePattern<mlir::stablehlo::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::stablehlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    ShapedType lhsTy = lhs.getType().cast<ShapedType>();
    ShapedType rhsTy = rhs.getType().cast<ShapedType>();
    ShapedType resultTy = op.getType().cast<ShapedType>();

    llvm::SmallVector<int64_t> dotShape;
    if (lhsTy.getRank() == 2 && lhsTy.getDimSize(0) == 1) {
      lhs = b.create<mlir::stablehlo::ReshapeOp>(
          lhsTy.clone({lhsTy.getDimSize(1)}), lhs);
    } else if (lhsTy.getRank() == 2) {
      dotShape.push_back(lhsTy.getDimSize(0));
    }

    if (rhsTy.getRank() == 2 && rhsTy.getDimSize(1) == 1) {
      rhs = b.create<mlir::stablehlo::ReshapeOp>(
          rhsTy.clone({rhsTy.getDimSize(0)}), rhs);
    } else if (rhsTy.getRank() == 2) {
      dotShape.push_back(rhsTy.getDimSize(1));
    }

    if (lhs == op.getLhs() && rhs == op.getRhs()) {
      return rewriter.notifyMatchFailure(op, "no vector reform available.");
    }

    auto newDot = b.create<mlir::stablehlo::DotOp>(
        resultTy.clone(dotShape), lhs, rhs, op.getPrecisionConfigAttr());
    auto resultReshape = b.create<mlir::stablehlo::ReshapeOp>(resultTy, newDot);

    rewriter.replaceOp(op, resultReshape);
    return success();
  }
};

struct DotGeneralToDot final : impl::DotGeneralToDotBase<DotGeneralToDot> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePreprocessingDotGeneralToDotPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populatePreprocessingDotGeneralToDotPatterns(mlir::MLIRContext *context,
                                                  RewritePatternSet *patterns,
                                                  PatternBenefit benefit) {
  patterns
      ->add<GeneralDotConvert, GeneralDotRemoveBatch, DotVectorOptimization>(
          context, benefit);
}

} // namespace mlir::iree_compiler::stablehlo
