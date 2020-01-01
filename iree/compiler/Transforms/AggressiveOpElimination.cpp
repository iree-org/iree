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

#include <deque>
#include <memory>

#include "iree/compiler/IR/Interpreter/HLOps.h"
#include "iree/compiler/IR/Interpreter/LLOps.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

template <typename T>
struct EraseUnused : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(T op,
                                     PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return this->matchSuccess();
    }
    return this->matchFailure();
  }
};

void populateAggressiveOpEliminationPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *ctx) {
  patterns.insert<EraseUnused<LoadOp>, EraseUnused<AllocOp>,
                  EraseUnused<IREEInterp::HL::AllocHeapOp>,
                  EraseUnused<IREEInterp::LL::AllocHeapOp>>(ctx);
}

}  // namespace

// TODO(b/142012496) Make these be handled by normal DCE.
class AggressiveOpEliminationPass
    : public FunctionPass<AggressiveOpEliminationPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateAggressiveOpEliminationPatterns(patterns, &getContext());

    applyPatternsGreedily(getFunction(), patterns);
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createAggressiveOpEliminationPass() {
  return std::make_unique<AggressiveOpEliminationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
