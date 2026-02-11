// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/TOSA/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tosa;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TOSATOLINALGEXTPASS
#include "compiler/plugins/input/TOSA/InputConversion/Passes.h.inc"

namespace {

// Converts tosa.scatter to the iree_linalg_ext.scatter operation. As the
// LinalgExt version is not batched therefore we materialize the batch index
// for each update.
class ScatterConversion : public OpRewritePattern<tosa::ScatterOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(tosa::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto values = op.getValuesIn();
    auto indices = cast<Value>(op.getIndices());
    auto updates = cast<Value>(op.getInput());
    auto valuesTy = dyn_cast<RankedTensorType>(values.getType());
    auto indicesTy = dyn_cast<RankedTensorType>(indices.getType());
    auto updatesTy = dyn_cast<RankedTensorType>(updates.getType());
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (!valuesTy || !indicesTy || !updatesTy) {
      return rewriter.notifyMatchFailure(op,
                                         "tosa.gather has unknown input rank");
    }

    // TOSA's scatter does not include a index dimension, instead it implicitly
    // supports an index depth of one. We materialize that implicit index of
    // one as follows: [batch, updates] -> [batch, updates, index_depth=1] With
    // a indexing map of [[0], [1, 2]].
    llvm::SmallVector<int64_t> expandIndShape{indicesTy.getDimSize(0),
                                              indicesTy.getDimSize(1), 1};
    SmallVector<ReassociationExprs> expandIndMap;
    expandIndMap.push_back({
        builder.getAffineDimExpr(0),
    });
    expandIndMap.push_back({
        builder.getAffineDimExpr(1),
        builder.getAffineDimExpr(2),
    });

    indices = tensor::ExpandShapeOp::create(
        builder, indicesTy.clone(expandIndShape), indices, expandIndMap);
    indicesTy = dyn_cast<RankedTensorType>(indices.getType());

    // Materialize the batch indice as LinalgExt scatter is not batched.
    {
      llvm::SmallVector<Value> dynDims;
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i) {
        if (indicesTy.isDynamicDim(i)) {
          dynDims.push_back(tensor::DimOp::create(builder, indices, i));
        }
      }

      Value empty = tensor::EmptyOp::create(
          builder, indicesTy.getShape(), indicesTy.getElementType(), dynDims);

      Value batchIdx = nullptr;

      if (indicesTy.getDimSize(0) == 1) {
        Value zero = arith::ConstantOp::create(
            builder, rewriter.getZeroAttr(indicesTy.getElementType()));
        batchIdx = linalg::FillOp::create(builder, zero, empty).getResult(0);
      } else {
        SmallVector<utils::IteratorType> iterators(
            indicesTy.getRank(), utils::IteratorType::parallel);
        SmallVector<AffineMap, 3> indexingMaps(
            2, builder.getMultiDimIdentityMap(indicesTy.getRank()));

        auto blockBuilder = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                                ValueRange blockArgs) {
          ImplicitLocOpBuilder b(op.getLoc(), nestedBuilder);
          auto index = linalg::IndexOp::create(b, 0);
          auto cast =
              arith::IndexCastOp::create(b, indicesTy.getElementType(), index);
          linalg::YieldOp::create(b, cast.getResult());
        };
        batchIdx =
            linalg::GenericOp::create(builder, indicesTy, indices, empty,
                                      indexingMaps, iterators, blockBuilder)
                .getResult(0);
      }

      indicesTy = cast<RankedTensorType>(indicesTy.clone(
          {indicesTy.getDimSize(0), indicesTy.getDimSize(1), 2}));
      indices = tosa::ConcatOp::create(builder, indicesTy,
                                       ValueRange{batchIdx, indices},
                                       rewriter.getI32IntegerAttr(2));
    }

    auto collapseBatch = [](Value value, ImplicitLocOpBuilder &b) -> Value {
      auto valueTy = cast<ShapedType>(value.getType());
      llvm::SmallVector<int64_t> collapseShape(valueTy.getShape().drop_front());
      llvm::SmallVector<ReassociationExprs> collapseMap(valueTy.getRank() - 1);
      collapseMap.front().push_back(b.getAffineDimExpr(0));
      for (int i = 0, s = collapseMap.size(); i < s; ++i) {
        collapseMap[i].push_back(b.getAffineDimExpr(i + 1));
      }

      int64_t batch = valueTy.getShape().front();
      int64_t rows = collapseShape.front();
      bool batchDyn = ShapedType::isDynamic(batch);
      bool rowsDyn = ShapedType::isDynamic(rows);
      collapseShape[0] =
          (batchDyn || rowsDyn) ? ShapedType::kDynamic : batch * rows;

      return tensor::CollapseShapeOp::create(b, valueTy.clone(collapseShape),
                                             value, collapseMap);
    };

    indices = collapseBatch(indices, builder);
    updates = collapseBatch(updates, builder);

    // Create the LinalgExt scatter operation.
    auto scatter = IREE::LinalgExt::ScatterOp::create(
        builder, TypeRange{values.getType()},
        /*updates=*/updates, /*indices=*/indices, /*original=*/values,
        builder.getDenseI64ArrayAttr({0, 1}), builder.getBoolAttr(true));

    llvm::SmallVector<Type> args(2, valuesTy.getElementType());
    Block *scatterBody =
        builder.createBlock(&scatter.getRegion(), {}, args,
                            llvm::SmallVector<Location>(2, op.getLoc()));
    builder.setInsertionPointToStart(scatterBody);
    IREE::LinalgExt::YieldOp::create(builder, scatterBody->getArgument(0));
    rewriter.replaceOp(op, scatter.getResult(0));
    return success();
  }
};

class TosaToLinalgExtPass final
    : public impl::TosaToLinalgExtPassBase<TosaToLinalgExtPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ScatterOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    mlir::FunctionOpInterface funcOp = getOperation();
    mlir::iree_compiler::populateTosaToLinalgExtPatterns(&patterns);
    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTosaToLinalgExtPatterns(RewritePatternSet *patterns) {
  patterns->add<ScatterConversion>(patterns->getContext());
}

} // namespace mlir::iree_compiler
