//===----------------------------------------------------------------------===//
// ConvertConvFilterToChannelsLastPass
//===----------------------------------------------------------------------===//

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
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
static AffineMap constructFilterMap(AffineMap map, SmallVector<int64_t> &perm) {
  unsigned numDims = map.getNumDims();
  ArrayRef<AffineExpr> mapResults = map.getResults();
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < perm.size(); ++i) {
    exprs.push_back(mapResults[perm[i]]);
  }
  return AffineMap::get(numDims, map.getNumSymbols(), exprs, map.getContext());
}

// Utility function to create a transpose operation.
static Value createTransposeOp(PatternRewriter &rewriter, Location loc,
                               Value tensor, SmallVector<int64_t> &perm) {
  SmallVector<OpFoldResult, 4> dimSizes =
      tensor::getMixedSizes(rewriter, loc, tensor);
  // Dim index of H, W, F, C
  SmallVector<OpFoldResult, 4> transposeResultDims;
  for (int i = 0; i < perm.size(); ++i) {
    transposeResultDims.push_back(dimSizes[perm[i]]);
  }

  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, transposeResultDims, tensorType.getElementType());
  return rewriter.create<linalg::TransposeOp>(loc, tensor, emptyTensor, perm)
      .getResult()[0];
}

llvm::LogicalResult convertConvFilterToTargetLayout(
    linalg::Conv2DNhwcHwcfOp convOp, PatternRewriter &rewriter,
    SmallVector<int64_t> &perm) {
  Location loc = convOp.getLoc();

  // Extract operands.
  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  // Extract indexing maps.
  AffineMap inputMap = convOp.getIndexingMapsArray()[0];
  AffineMap filterMap = convOp.getIndexingMapsArray()[1];
  AffineMap outputMap = convOp.getIndexingMapsArray()[2];

  AffineMap transposedFilterMap = constructFilterMap(filterMap, perm);
  Value transposedFilter = createTransposeOp(rewriter, loc, filter, perm);

  SmallVector<utils::IteratorType> iterators = convOp.getIteratorTypesArray();

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{input, transposedFilter}, output,
      ArrayRef<AffineMap>{inputMap, transposedFilterMap, outputMap},
      iterators);

  // Reuse the same payload as the original convolution op.
  rewriter.inlineRegionBefore(convOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());

  rewriter.replaceOp(convOp, genericOp->getResults());
  return success();
}

namespace {
struct ConvertHwcfToHwfc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {0, 1, 3, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
  }
};

struct ConvertHwcfToFhwc : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm = {3, 0, 1, 2};
    return convertConvFilterToTargetLayout(convOp, rewriter, perm);
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
    if (filterLayout == "hwfc") {
      patterns.add<ConvertHwcfToHwfc>(context);
    } else if (filterLayout == "fhwc") {
      patterns.add<ConvertHwcfToFhwc>(context);
    } else {
      // TODO add default fallback to filter layout once we have more data
      // about models with the two layouts
      llvm_unreachable("Unsupported filter layout. Use hwfc or fhwc.");
    }

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }

    LDBG("after converting convolutions to channels last\n" << *op);
  }

private:
  llvm::SmallString<4> layout;
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
