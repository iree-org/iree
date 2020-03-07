// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cstdint>
#include <memory>

#include "iree/compiler/Translation/XLAToLinalg/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

// These are duplicated from xla/transforms/xla_legalize_to_linalg.cc.
ArrayAttr getNParallelLoopsAttrs(unsigned nParallelLoops, Builder& b) {
  auto parallelLoopTypeAttr = b.getStringAttr("parallel");
  SmallVector<Attribute, 3> iteratorTypes(nParallelLoops, parallelLoopTypeAttr);
  return b.getArrayAttr(iteratorTypes);
}

ShapedType getXLAOpResultType(Operation* op) {
  return op->getResult(0).getType().cast<ShapedType>();
}

template <bool isLHLO = true>
bool verifyXLAOpTensorSemantics(Operation* op) {
  auto verifyType = [&](Value val) -> bool {
    return (val.getType().isa<RankedTensorType>());
  };
  return llvm::all_of(op->getOperands(), verifyType) &&
         llvm::all_of(op->getResults(), verifyType);
}

/// Conversion pattern for splat constants that are not scalars.
class SplatConstConverter : public OpConversionPattern<ConstantOp> {
 public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      ConstantOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!verifyXLAOpTensorSemantics(op)) {
      return matchFailure();
    }
    auto resultType = getXLAOpResultType(op);
    if (resultType.getRank() == 0) return matchFailure();
    auto valueAttr = op.value().template cast<DenseElementsAttr>();
    if (!valueAttr.isSplat()) return matchFailure();

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    auto nloops = std::max<unsigned>(resultType.getRank(), 1);
    auto loc = op.getLoc();

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, args, rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(1),
        rewriter.getAffineMapArrayAttr(rewriter.getMultiDimIdentityMap(nloops)),
        getNParallelLoopsAttrs(nloops, rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr,
        /*library_call=*/nullptr);
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    rewriter.setInsertionPointToEnd(block);
    auto stdConstOp =
        rewriter.create<mlir::ConstantOp>(loc, valueAttr.getSplatValue());
    rewriter.create<linalg::YieldOp>(loc, stdConstOp.getResult());
    rewriter.replaceOp(op, linalgOp.getResults());
    return matchSuccess();
  }
};

// Reshape by collapsing or expanding consecutive dims e.g (3, 2, 4) -> (3, 8),
// (12, 3) -> (3, 4, 3).
class ReshapeExpandOrCollapse : public OpConversionPattern<xla_hlo::ReshapeOp> {
 public:
  using OpConversionPattern<xla_hlo::ReshapeOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_hlo::ReshapeOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto operandType = op.operand().getType().cast<ShapedType>();
    auto resultType = op.getResult().getType().cast<ShapedType>();
    if (!operandType.getRank()) return matchFailure();

    auto srcDims = operandType.getShape();
    auto srcRank = operandType.getRank();

    auto dstDims = resultType.getShape();
    auto dstRank = resultType.getRank();

    SmallVector<AffineExpr, 2> inputExprs;

    // This is always an expand dims where srcRank < dstRank.
    auto findRange =
        [](const ArrayRef<int64_t> srcShape,
           const ArrayRef<int64_t> dstShape) -> std::pair<int64_t, int64_t> {
      const std::pair<int64_t, int64_t> invalidRange = {-1, -1};
      int64_t start = 0, end = 0;
      int64_t srcRank = srcShape.size();
      int64_t dstRank = dstShape.size();
      while (start < srcRank && srcShape[start] == dstShape[start]) start++;
      if (start >= srcRank) return invalidRange;
      int64_t size = srcShape[start];
      int64_t dstSize = 1;
      end = start;
      while (end < dstRank && dstSize < size) dstSize *= dstShape[end++];
      if (end > dstRank) return invalidRange;
      return {start, end - 1};
    };

    // Return strides in the range[start, end]
    auto getRangeStrides = [](const ArrayRef<int64_t> dims, int start,
                              int end) -> llvm::SmallVector<int64_t, 4> {
      int rangeSize = end - start + 1;
      llvm::SmallVector<int64_t, 4> strides(rangeSize, 1);
      strides[rangeSize - 1] = 1;
      for (int i = rangeSize - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[start + i + 1];
      }
      return strides;
    };
    // ExpandDim
    if (dstRank > srcRank) {
      auto range = findRange(srcDims, dstDims);
      if (range.second == -1 || range.first == -1) return matchFailure();
      // Don't match expand by ones.
      for (int i = range.first; i <= range.second; ++i) {
        if (dstDims[i] == srcDims[range.first]) return matchFailure();
      }
      auto strides = getRangeStrides(dstDims, range.first, range.second);
      int dimSize = range.second - range.first;
      for (int i = 0; i < srcRank; ++i) {
        if (i == range.first) {
          auto dimExpr = rewriter.getAffineDimExpr(i) * strides[0];
          for (int j = 1; j <= dimSize; ++j) {
            dimExpr = dimExpr + rewriter.getAffineDimExpr(i + j) * strides[j];
          }
          inputExprs.push_back(dimExpr);
        } else if (i > range.first) {
          inputExprs.push_back(rewriter.getAffineDimExpr(i + dimSize));
        } else {
          inputExprs.push_back(rewriter.getAffineDimExpr(i));
        }
      }
    } else if (srcRank > dstRank) {
      // CollapseDim
      auto range = findRange(dstDims, srcDims);
      if (range.first == -1 || range.second == -1) return matchFailure();
      // Don't match expand by ones.
      for (int i = range.first; i <= range.second; ++i) {
        if (srcDims[i] == dstDims[range.first]) return matchFailure();
      }
      auto strides = getRangeStrides(srcDims, range.first, range.second);
      int dimSize = range.second - range.first;
      for (int i = 0; i < srcRank; ++i) {
        if (i < range.first) {
          inputExprs.push_back(rewriter.getAffineDimExpr(i));
        } else if (i > range.second) {
          const auto index = i - dimSize;
          index < dstRank
              ? inputExprs.push_back(rewriter.getAffineDimExpr(index))
              : inputExprs.push_back(rewriter.getAffineConstantExpr(0));

        } else {
          inputExprs.push_back(rewriter.getAffineDimExpr(range.first)
                                   .floorDiv(strides[i - range.first]) %
                               srcDims[i]);
        }
      }
    } else {
      return matchFailure();
    }
    auto srcMap = AffineMap::get(dstRank, /*symbolCount=*/0, inputExprs);
    auto dstMap = rewriter.getMultiDimIdentityMap(dstRank);

    auto indexingMapsAttr = rewriter.getAffineMapArrayAttr({srcMap, dstMap});

    OpBuilder::InsertionGuard linalgOpGuard(rewriter);
    auto nloops = resultType.getRank();
    auto loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, args, rewriter.getI64IntegerAttr(1),
        rewriter.getI64IntegerAttr(1), indexingMapsAttr,
        getNParallelLoopsAttrs(nloops, rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(operandType.getElementType());

    rewriter.setInsertionPointToEnd(block);
    rewriter.create<linalg::YieldOp>(loc, block->getArgument(0));

    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return matchSuccess();
  }
};

struct XlaLegalizeToLinalg : public FunctionPass<XlaLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) -> bool {
      return isa<linalg::LinalgOp>(op.getOperation()->getParentOp());
    });

    auto func = getFunction();
    populateXlaToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateXlaToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns) {
  xla_hlo::populateHLOToLinalgConversionPattern(context, patterns);
  patterns->insert<SplatConstConverter, ReshapeExpandOrCollapse>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgPass() {
  return std::make_unique<XlaLegalizeToLinalg>();
}

static PassRegistration<XlaLegalizeToLinalg> legalize_pass(
    "iree-hlo-to-linalg", "Legalize from HLO dialect to Linalg dialect");

}  // namespace iree_compiler
}  // namespace mlir
