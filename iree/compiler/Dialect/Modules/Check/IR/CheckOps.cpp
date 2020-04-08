// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

namespace {
template <typename SrcOp, typename DstOp>
struct ExpandAttributeToConst : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto rhs = rewriter.create<ConstantOp>(op.getLoc(), op.value());
    rewriter.replaceOpWithNewOp<DstOp>(op, op.lhs(), rhs);
    return success();
  }
};
}  // namespace

void ExpectEqConstOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandAttributeToConst<ExpectEqConstOp, ExpectEqOp>>(context);
}

void ExpectAlmostEqConstOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results
      .insert<ExpandAttributeToConst<ExpectAlmostEqConstOp, ExpectAlmostEqOp>>(
          context);
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.cpp.inc"

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
