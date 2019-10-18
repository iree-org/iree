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

#include "iree/compiler/IR/Dialect.h"
#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ConstantOp op,
                                     PatternRewriter &rewriter) const override {
    if (auto elementsValue = op.getValue().dyn_cast<ElementsAttr>()) {
      auto ireeConst =
          rewriter.create<IREE::ConstantOp>(op.getLoc(), elementsValue);

      auto result = wrapAsTensor(ireeConst.getResult(), op, rewriter);
      rewriter.replaceOp(op, result);
      return matchSuccess();
    }

    auto type = op.getValue().getType();
    if (!type.isIntOrFloat()) {
      return matchFailure();
    }
    auto elementsValue =
        DenseElementsAttr::get(RankedTensorType::get({}, type), op.getValue());
    auto ireeConst =
        rewriter.create<IREE::ConstantOp>(op.getLoc(), elementsValue);
    rewriter.replaceOpWithNewOp<IREE::MemRefToScalarOp>(op, ireeConst);
    return matchSuccess();
  }
};

struct ExtractElementOpLowering : public OpRewritePattern<ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ExtractElementOp op,
                                     PatternRewriter &rewriter) const override {
    Value *memRefInput =
        wrapAsMemRef(loadAccessValue(op.getLoc(), op.getAggregate(), rewriter),
                     op, rewriter);

    SmallVector<Value *, 4> indices = {op.indices().begin(),
                                       op.indices().end()};
    rewriter.replaceOpWithNewOp<LoadOp>(op, memRefInput, indices);
    return matchSuccess();
  }
};

}  // namespace

void populateLowerStdToIreePatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *ctx) {
  patterns.insert<ConstantOpLowering, ExtractElementOpLowering>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
