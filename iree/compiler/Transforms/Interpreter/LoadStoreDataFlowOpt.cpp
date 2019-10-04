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

#include <algorithm>

#include "iree/compiler/IR/Interpreter/HLOps.h"
#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "iree/compiler/Utils/OpCreationUtils.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LoadStoreToCopy : public OpRewritePattern<StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(StoreOp storeOp,
                                     PatternRewriter &rewriter) const override {
    auto loadOp = dyn_cast<LoadOp>(storeOp.getValueToStore()->getDefiningOp());
    if (!loadOp) {
      return matchFailure();
    }
    if (loadOp.getMemRefType().getRank() != 0 ||
        storeOp.getMemRefType().getRank() != 0) {
      // TODO(b/138851470) Support non-scalar folding
      return matchFailure();
    }

    auto emptyArrayMemref = createArrayConstant(rewriter, storeOp.getLoc(), {});
    rewriter.create<IREEInterp::HL::CopyOp>(
        storeOp.getLoc(), loadOp.getMemRef(),
        /*srcIndices=*/emptyArrayMemref, storeOp.getMemRef(),
        /*dstIndices=*/emptyArrayMemref, /*lengths=*/emptyArrayMemref);
    storeOp.erase();
    return matchSuccess();
  }
};

// TODO(b/141771852) Figure out how to have the same pattern delete the load and
// store
struct EraseUnusedLoad : public OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(LoadOp loadOp,
                                     PatternRewriter &rewriter) const override {
    if (loadOp.use_empty()) {
      loadOp.erase();
      return matchSuccess();
    }
    return matchFailure();
  }
};

}  // namespace

class InterpreterLoadStoreDataFlowOptPass
    : public FunctionPass<InterpreterLoadStoreDataFlowOptPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<LoadStoreToCopy, EraseUnusedLoad>(&getContext());

    // TODO(b/141771852) Incorporate these patterns into dialect conversion
    // instead?
    applyPatternsGreedily(getFunction(), patterns);
  }
};

std::unique_ptr<OpPassBase<FuncOp>>
createInterpreterLoadStoreDataFlowOptPass() {
  return std::make_unique<InterpreterLoadStoreDataFlowOptPass>();
}

static PassRegistration<InterpreterLoadStoreDataFlowOptPass> pass(
    "iree-interpreter-load-store-data-flow-opt",
    "Optimize local load and store data flow by removing redundant accesses");

}  // namespace iree_compiler
}  // namespace mlir
