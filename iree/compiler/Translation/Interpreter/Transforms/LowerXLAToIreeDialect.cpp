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

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/IR/CommonOps.h"
#include "iree/compiler/Translation/Interpreter/Utils/MemRefUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstOpLowering : public OpRewritePattern<xla_hlo::ConstOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(xla_hlo::ConstOp op,
                                     PatternRewriter &rewriter) const override {
    auto ireeConst =
        rewriter.create<IREEInterp::ConstantOp>(op.getLoc(), op.value());
    rewriter.replaceOp(op, wrapAsTensor(ireeConst, op, rewriter));
    return matchSuccess();
  }
};

}  // namespace

void populateLowerXlaToIreePatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *ctx) {
  patterns.insert<ConstOpLowering>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
