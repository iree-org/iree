//===----------------------------------------------------------------------===//
// ConvertConvFilterToChannelsLastPass
//===----------------------------------------------------------------------===//

#include <cstdint>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-convert-conv-filter-to-channels-last"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_CONVERTCONVFILTERTOCHANNELSLASTPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

// Utility function to swap the last two dimensions.
static AffineMap swapLastTwoDims(AffineMap map) {
  map.dump();
  unsigned numDims = map.getNumDims();
  unsigned mapSize = map.getNumResults();
  SmallVector<AffineExpr> exprs(map.getResults().begin(),
                                map.getResults().end());
  std::swap(exprs[mapSize - 2], exprs[mapSize - 1]);
  return AffineMap::get(numDims, map.getNumSymbols(), exprs, map.getContext());
}

// Utility function to create a transpose operation.
static Value createTransposeOp(PatternRewriter &rewriter, Location loc,
                               Value tensor, AffineMap transposedMap) {
  SmallVector<OpFoldResult> dimSizes =
      tensor::getMixedSizes(rewriter, loc, tensor);
  SmallVector<OpFoldResult> transposeResultDims = {dimSizes[0], dimSizes[1],
                                                   dimSizes[3], dimSizes[2]};

  SmallVector<int64_t, 4> transposedShape = {0, 1, 3, 2};
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, transposeResultDims, tensorType.getElementType());
  return rewriter
      .create<linalg::TransposeOp>(loc, tensor, emptyTensor, transposedShape)
      .getResult()[0];
}

namespace {
struct ConvertLinalgConvNhwcHwcf
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Extract operands.
    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    // Extract indexing maps.
    AffineMap inputMap = convOp.getIndexingMapsArray()[0];
    AffineMap filterMap = convOp.getIndexingMapsArray()[1];
    AffineMap outputMap = convOp.getIndexingMapsArray()[2];

    // Swap the last two dimensions (C <-> F) for the filter.
    AffineMap transposedFilterMap = swapLastTwoDims(filterMap);

    // Create a transposed filter tensor.
    Value transposedFilter =
        createTransposeOp(rewriter, loc, filter, transposedFilterMap);

    SmallVector<utils::IteratorType> iterators = convOp.getIteratorTypesArray();

    auto linalgGenericOp = rewriter.create<linalg::GenericOp>(
        loc, output.getType(), ValueRange{input, transposedFilter}, output,
        ArrayRef<AffineMap>{inputMap, transposedFilterMap, outputMap},
        iterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          // Block arguments correspond to input, filter, and output.
          Value inputElem = blockArgs[0];
          Value filterElem = blockArgs[1];
          filterElem = rewriter.create<arith::ExtFOp>(
              nestedLoc, rewriter.getF32Type(), filterElem);
          inputElem = rewriter.create<arith::ExtFOp>(
              nestedLoc, rewriter.getF32Type(), inputElem);

          Value acc = blockArgs[2];
          Value result = nestedBuilder.create<arith::AddFOp>(
              nestedLoc, acc,
              nestedBuilder.create<arith::MulFOp>(nestedLoc, inputElem,
                                                  filterElem));

          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(convOp, linalgGenericOp->getResults());
    return success();
  }
};

class ConvertConvFilterToChannelsLastPass
    : public iree_compiler::Preprocessing::impl::
          ConvertConvFilterToChannelsLastPassBase<
              ConvertConvFilterToChannelsLastPass> {
public:
  using iree_compiler::Preprocessing::impl::
      ConvertConvFilterToChannelsLastPassBase<
          ConvertConvFilterToChannelsLastPass>::
          ConvertConvFilterToChannelsLastPassBase;

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvertLinalgConvNhwcHwcf>(context);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }

    LDBG("after converting convolutions to channels last\n" << *op);
  }
};

} // namespace

} // namespace mlir::iree_compiler::Preprocessing
