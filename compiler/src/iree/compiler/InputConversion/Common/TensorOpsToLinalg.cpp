
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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::InputConversion {

struct GatherToLinalg : public OpRewritePattern<tensor::GatherOp> {
  using OpRewritePattern<tensor::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType();
    auto gatherDims = op.getGatherDims();

    // not sure if this is necessary
    assert(op.getIndices().getType().getShape().back() == gatherDims.size() &&
           "Last dimension of result type must match number of gather dims");

    // create affine map for indices
    // we want to loop over all diminesions but the least significant one
    // since this is the dimension that contains the indices
    // so we create a index map for every "gather_dim", the index maps will be
    // used to access the indices at the given dimension
    llvm::SmallVector<mlir::AffineMap, 2> indexingMaps;

    for (int i = 0; i < gatherDims.size(); i++) {
      llvm::SmallVector<AffineExpr, 3> exprs;
      for (int i = 0; i < op.getIndicesType().getRank() - 1; i++) {
        exprs.push_back(rewriter.getAffineDimExpr(i));
      }
      exprs.push_back(getAffineConstantExpr(i, rewriter.getContext()));
      indexingMaps.push_back(mlir::AffineMap::get(
          resultType.getRank(), 0, exprs, rewriter.getContext()));
    }

    Value outTensor;
    if (!resultType.hasStaticShape()) {
      llvm::SmallVector<Value> dynamicDims;
      for (int i = 0; i < op.getIndicesType().getRank(); i++) {
        if (ShapedType::isDynamic(resultType.getShape()[i])) {
          Value c = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
          dynamicDims.push_back(c);
        }
      }
      outTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), resultType.getShape(), resultType.getElementType(),
          dynamicDims);
    } else {
      outTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), resultType.getShape(), resultType.getElementType());
    }

    auto outputMap = rewriter.getMultiDimIdentityMap(resultType.getRank());

    indexingMaps.push_back(outputMap);

    llvm::SmallVector<mlir::utils::IteratorType, 3> iteratorTypes;
    for (int i = 0; i < resultType.getRank(); i++) {
      iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
    }

    // duplicate the indices for every index map we have created
    // since each one will reference a different constant
    llvm::SmallVector<Value, 2> ins;
    for (int i = 0; i < gatherDims.size(); i++) {
      ins.push_back(op.getIndices());
    }

    int numLeadingDimsFromIndices = op.getIndices().getType().getRank() - 1;

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultType, ins, outTensor, indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // assume sources will be a tensor of rank x and indices will
          // be a tensor of rank y, so the resulting tensor will be of
          // of rank  y - 1 + (num dimensions not gathered)
          // so the first y - 1 loop nests will be responsible for looping over
          // the indices while  the remaining x will be either loaded(gathered)
          // by the indices tensor which makes their dimension 1 (or can be
          // omitted) or they will be copied from the source tensor


          llvm::SmallVector<Value> indices(op.getSource().getType().getRank());

          // first set gathered indices from the indices tensor
          for (int i = 0; i < gatherDims.size(); i++) {
            Value idx = nestedBuilder.create<arith::IndexCastOp>(
                nestedLoc, rewriter.getIndexType(), args[i]);
            indices[gatherDims[i]] = idx;
          }

          // collect remaining dims that are to be copied from the source tensor
          // pair<sourceDim, dimSize>
          llvm::SmallVector<std::pair<int64_t, int64_t>> dimsFromSource;
          for (int i = 0; i < indices.size(); i++) {
            if (!indices[i])
              dimsFromSource.emplace_back(
                  i, op.getSource().getType().getShape()[i]);
          }

          // pick the correct linalg.index from where to copy the dims
          if (!dimsFromSource.empty()) {
            auto resultShape = resultType.getShape();
            int sourceDimIdx = 0;
            for (int i = numLeadingDimsFromIndices; i < resultShape.size();
                 i++) {
              if (resultShape[i] == dimsFromSource[sourceDimIdx].second) {
                indices[dimsFromSource[sourceDimIdx].first] =
                    nestedBuilder.create<linalg::IndexOp>(nestedLoc, i);
                sourceDimIdx++;
              }
            }
          }

          Value extracted = nestedBuilder.create<tensor::ExtractOp>(
              nestedLoc, resultType.getElementType(), op.getSource(), indices);
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
