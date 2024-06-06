
#include <cstdint>
#include <utility>
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::InputConversion {

struct GatherToLinalg : public OpRewritePattern<tensor::GatherOp> {
  using OpRewritePattern<tensor::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = op.getType();
    auto resultRank = resultType.getRank();
    auto indicesType = op.getIndicesType();
    auto indicesRank = indicesType.getRank();
    auto gatherDims = op.getGatherDims();
    auto sourceRank = op.getSourceType().getRank();

    // not sure if this is necessary
    assert(op.getIndices().getType().getShape().back() == gatherDims.size() &&
           "Last dimension of result type must match number of gather dims");

    Value outTensor;

    if (!resultType.hasStaticShape()) {
      llvm::SmallVector<Value> dynamicDims;
      for (int i = 0; i < indicesRank; i++) {
        if (ShapedType::isDynamic(resultType.getShape()[i])) {
          Value c = rewriter.create<arith::ConstantIndexOp>(loc, i);
          dynamicDims.push_back(c);
        }
      }
      outTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType(),
          dynamicDims);
    } else {
      outTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType());
    }

    SmallVector<Value> constants;
    for (int64_t i = 0; i < gatherDims.size(); i++){
      constants.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, i));
    }

    SmallVector<AffineMap, 1> indexingMaps = {rewriter.getMultiDimIdentityMap(resultRank)};
    llvm::SmallVector<mlir::utils::IteratorType, 3> iteratorTypes;
    for (int64_t i = 0; i < resultRank; i++) {
      iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ValueRange{}, outTensor, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // assume sources will be a tensor of rank x and indices will
          // be a tensor of rank y, so the resulting tensor will be of
          // of rank  y - 1 + (num dimensions not gathered)
          // so the first y - 1 loop nests will be responsible for looping over
          // the indices while the remaining x will be either loaded(gathered)
          // by the indices tensor which makes their dimension 1 (or can be
          // omitted) or they will be copied from the source tensor


          // create linalg indices here to avoid duplicates, might create some
          // indices that are not used
          SmallVector<Value> linalgIndices;
          for (int64_t i = 0; i < resultRank; ++i) {
            linalgIndices.push_back(rewriter.create<linalg::IndexOp>(nestedLoc, i));
          }
          // create one extract op for every gather dim that loads the index
          // from the indices tensor
          SmallVector<Value> extractedIndices;
          for (int64_t i = 0; i < gatherDims.size(); i++) {
            // the first indicesRank - 1 linalg.index operation correspond to
            // the batch dimension of the indices tensor
            SmallVector<Value> indicesForIndices;
            for (int64_t j = 0; j < indicesRank - 1; j++) {
              indicesForIndices.push_back(linalgIndices[j]);
            }
            indicesForIndices.push_back(constants[i]);
            Value extractedIndex = nestedBuilder.create<tensor::ExtractOp>(
                nestedLoc, indicesType.getElementType(), op.getIndices(),
                indicesForIndices);
            Value castedIndex = nestedBuilder.create<arith::IndexCastOp>(nestedLoc, nestedBuilder.getIndexType(), extractedIndex);
            extractedIndices.push_back(castedIndex);
          }


          // set the loaded indices to the corresponding gatherDims
          SmallVector<Value> indicesForSource(sourceRank);
          for (int64_t i = 0; i < extractedIndices.size(); i++) {
            indicesForSource[gatherDims[i]] = extractedIndices[i];
          }

          // collect remaining dims that are to be copied from the source tensor
          // pair<sourceDim, dimSize>
          llvm::SmallVector<std::pair<int64_t, int64_t>> dimsFromSource;
          for (int i = 0; i < indicesForSource.size(); i++) {
            indicesForSource[i].dump();
            if (!indicesForSource[i])
              dimsFromSource.emplace_back(
                  i, op.getSource().getType().getShape()[i]);
          }

          // pick the correct linalg.index from where to copy the dims
          if (!dimsFromSource.empty()) {
            auto resultShape = resultType.getShape();
            int64_t sourceDimIdx = 0;
            for (int64_t i = indicesRank - 1; i < resultRank; i++) {
              // if size of the resultDim and SourceDim are the same we found a
              // match, assign the linalg index to the corresponding axis
              if (resultShape[i] == dimsFromSource[sourceDimIdx].second) {
                indicesForSource[dimsFromSource[sourceDimIdx].first] = linalgIndices[i];
                sourceDimIdx++;
              }
            }
            assert(sourceDimIdx == dimsFromSource.size() && "not all source indices have been assigned");
          }

          Value extracted = nestedBuilder.create<tensor::ExtractOp>(
              nestedLoc, resultType.getElementType(), op.getSource(), indicesForSource);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, extracted);
        });

    // linalgOp.dump();
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
