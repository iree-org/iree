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
  patterns->insert<SplatConstConverter>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgPass() {
  return std::make_unique<XlaLegalizeToLinalg>();
}

static PassRegistration<XlaLegalizeToLinalg> legalize_pass(
    "iree-hlo-to-linalg", "Legalize from HLO dialect to Linalg dialect");

}  // namespace iree_compiler
}  // namespace mlir
