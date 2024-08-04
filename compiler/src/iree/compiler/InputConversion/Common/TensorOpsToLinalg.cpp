
#include <cstdint>
#include <functional>
#include <utility>
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::InputConversion {

struct DimInfo {
  int64_t sourceIdx;
  int64_t dimSize;
};

void matchDims(int64_t resStart, llvm::ArrayRef<int64_t> resultShape,
               llvm::ArrayRef<DimInfo> sourceIndices,
               std::function<void(int64_t resIdx, int64_t sourceIdx)> onMatch) {
  int64_t sourceIdx = 0;
  int64_t resIdx = resStart;
  while (resIdx < resultShape.size() && sourceIdx < sourceIndices.size()) {
    if (resultShape[resIdx] == sourceIndices[sourceIdx].dimSize) {
      onMatch(resIdx, sourceIndices[sourceIdx].sourceIdx);
      sourceIdx++;
      resIdx++;
    } else {
      resIdx++;
    }
  }
  assert(sourceIdx == sourceIndices.size() && "Could not match all dimensions");
}

struct GatherToLinalg : public OpRewritePattern<tensor::GatherOp> {
  using OpRewritePattern<tensor::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = op.getType();
    auto resultRank = resultType.getRank();
    auto resultShape = resultType.getShape();
    auto indicesType = op.getIndicesType();
    auto indicesRank = indicesType.getRank();
    auto indicesShape = indicesType.getShape();
    auto gatherDims = op.getGatherDims();
    auto sourceRank = op.getSourceType().getRank();

    // not sure if this is necessary
    assert(indicesShape.back() == gatherDims.size() &&
           "Last dimension of result type must match number of gather dims");

    // collect dims which are are not copied from the source tensor
    // its used later for matching dynamic dims and assigning linalg indices
    // in the inner loop
    llvm::SmallVector<DimInfo> dimsFromSource;
    // gather dims have to be ascending so we cant walk through both side by
    // side
    int gatherDimIdx = 0;
    for (int i = 0; i < sourceRank; i++) {
      if (gatherDimIdx < gatherDims.size() && i == gatherDims[gatherDimIdx]) {
        gatherDimIdx++;
        continue;
      }
      dimsFromSource.push_back({i, op.getSourceType().getShape()[i]});
    }

    // if the shape is not static we have to find the dynamic dimensions
    // dimsFromSource the indices and source tensor
    Value outTensor;
    if (!resultType.hasStaticShape()) {
      // find dynamic dims from the indices tensor
      llvm::SmallVector<Value> dynamicDims;
      for (int i = 0; i < indicesRank - 1; i++) {
        if (ShapedType::isDynamic(indicesShape[i]) &&
            ShapedType::isDynamic(resultShape[i])) {
          Value dim = rewriter.create<tensor::DimOp>(loc, op.getIndices(), i);
          dynamicDims.push_back(dim);
        }
      }
      // find dynamic dims from the source tensor
      matchDims(indicesRank - 1, resultShape, dimsFromSource,
                [&](int64_t resIdx, int64_t sourceIdx) {
                  if (ShapedType::isDynamic(resultShape[resIdx])) {
                    dynamicDims.push_back(rewriter.create<tensor::DimOp>(
                        loc, op.getSource(), sourceIdx));
                  }
                });

      outTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType(), dynamicDims);
    } else {
      outTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    }

    // create affine map for indices
    // we want to loop over all diminesions but the least significant one
    // since this is the dimension that contains the indices
    // so we create a index map for every "gather_dim", the index maps will be
    // used to access the indices at the given dimension
    llvm::SmallVector<mlir::AffineMap, 3> indexingMaps;

    for (int i = 0; i < gatherDims.size(); i++) {
      llvm::SmallVector<AffineExpr, 3> exprs;
      for (int i = 0; i < op.getIndicesType().getRank() - 1; i++) {
        exprs.push_back(rewriter.getAffineDimExpr(i));
      }
      exprs.push_back(getAffineConstantExpr(i, rewriter.getContext()));
      indexingMaps.push_back(mlir::AffineMap::get(
          resultType.getRank(), 0, exprs, rewriter.getContext()));
    }

    // output map
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    llvm::SmallVector<mlir::utils::IteratorType, 3> iteratorTypes;
    for (int i = 0; i < resultRank; i++) {
      iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    }

    // duplicate the indices for every index map we have created
    // since each one will reference a different constant
    llvm::SmallVector<Value, 3> ins;
    for (int i = 0; i < gatherDims.size(); i++) {
      ins.push_back(op.getIndices());
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ins, outTensor, indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // create linalg indices here to avoid duplicates, might create some
          // indices that are not used
          SmallVector<Value> linalgIndices;
          for (int64_t i = 0; i < resultRank; ++i) {
            linalgIndices.push_back(
                rewriter.create<linalg::IndexOp>(nestedLoc, i));
          }

          // set the loaded indices to the corresponding gatherDims
          SmallVector<Value> indicesForSource(sourceRank);
          for (int64_t i = 0; i < gatherDims.size(); i++) {
            Value idx = nestedBuilder.create<arith::IndexCastOp>(
                nestedLoc, rewriter.getIndexType(), args[i]);
            indicesForSource[gatherDims[i]] = idx;
          }

          // pick the correct linalg.index from where to copy the dims
          if (!dimsFromSource.empty()) {
            matchDims(indicesRank - 1, resultShape, dimsFromSource,
                      [&](int64_t resIdx, int64_t sourceIdx) {
                        indicesForSource[sourceIdx] = linalgIndices[resIdx];
                      });
          }

          Value extracted = nestedBuilder.create<tensor::ExtractOp>(
              nestedLoc, resultType.getElementType(), op.getSource(),
              indicesForSource);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, extracted);
        });

    rewriter.replaceAllUsesWith(op.getResult(), linalgOp.getResult(0));
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

struct TensorOpsToLinalgPass
    : public TensorOpsToLinalgPassBase<TensorOpsToLinalgPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<GatherToLinalg>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTensorOpsToLinalgPass() {
  return std::make_unique<TensorOpsToLinalgPass>();
}

} // namespace mlir::iree_compiler::InputConversion
