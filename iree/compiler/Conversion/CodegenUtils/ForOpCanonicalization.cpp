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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Pass to combine instructions across ForOp boundary. It is common when doing
// incremental lowering to generate transient ops that cancel each others out.
// Canonicalization usually clean up those operations. When the value is loop
// carried, MLIR canonicalization currently doesn't remove the redundant
// operations.
//
// This pass allow to workaround MLIR limitation and does ad hoc clean up of
// instructions found in IREE. Once we have a more general mechanism in MLIR
// this pass can be completely removed.
// This pass does this kind of transformation:
// ```
// %21 = vector.shape_cast %20 : vector<4xf32> to vector<1x4xf32>
// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %21)
//    -> vector<1x4xf32> {
//    [...]
//    %100 = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
//    [...]
//    %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
//    scf.yield %109 : vector<1x4xf32>
//  }
//  %24 = vector.shape_cast %22 : vector<1x4xf32> to vector<4xf32>
// ```
// ->
// ```
// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %20)
//    -> vector<4xf32> {
//    [...]
//    scf.yield %108 : vector<4xf32>
//  }
// ```

namespace mlir {
namespace iree_compiler {

namespace {
class ForOpArgFolding final : public OpRewritePattern<scf::ForOp> {
 public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  Value FoldCarryDep(scf::ForOp forOp, Operation* ivUser,
                     Operation* ivDef) const {
    if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(ivUser)) {
      if (auto souceOp = dyn_cast<vector::ShapeCastOp>(ivDef)) {
        if (shapeCast.getType() == souceOp.source().getType())
          return souceOp.source();
      }
    } else if (auto extractOp = dyn_cast<vector::ExtractOp>(ivUser)) {
      if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(ivDef)) {
        if (extractOp.getType() == broadcastOp.getSourceType())
          return broadcastOp.source();
      }
    }
    return Value();
  }

  void transferBody(Block* source, Block* dest, ArrayRef<Value> results,
                    PatternRewriter& rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    yieldOp.getOperation()->setOperands(results);
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<unsigned, 8> iteratorFolded;
    SmallVector<Operation*, 8> resultOps;
    auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto returnValues = llvm::to_vector<8>(terminator.getOperands());
    auto initArgs = llvm::to_vector<8>(forOp.getIterOperands());
    for (auto it : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!it.value().hasOneUse()) continue;
      Operation* op = it.value().use_begin()->getOwner();
      if (!isa<vector::ShapeCastOp, vector::ExtractOp>(op)) continue;
      Operation* returnValDef = returnValues[it.index()].getDefiningOp();
      Value newReturn = FoldCarryDep(forOp, op, returnValDef);
      if (!newReturn) continue;
      iteratorFolded.push_back(it.index());
      resultOps.push_back(returnValDef);
      returnValues[it.index()] = newReturn;

      BlockAndValueMapping mapping;
      mapping.map(it.value(), initArgs[it.index()]);
      initArgs[it.index()] = rewriter.clone(*op, mapping)->getResult(0);
    }
    if (iteratorFolded.empty()) return success();
    auto newLoop =
        rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.lowerBound(),
                                    forOp.upperBound(), forOp.step(), initArgs);
    transferBody(forOp.getBody(), newLoop.getBody(), returnValues, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 8> repResults(newLoop.getResults().begin(),
                                     newLoop.getResults().end());
    for (auto en : llvm::enumerate(iteratorFolded)) {
      BlockAndValueMapping mapping;
      mapping.map(returnValues[en.value()], newLoop.getResult(en.value()));
      repResults[en.index()] =
          rewriter.clone(*resultOps[en.index()], mapping)->getResult(0);
      Operation* oldOp =
          newLoop.getRegionIterArgs()[en.index()].use_begin()->getOwner();
      SmallVector<Value, 1> arg(1, newLoop.getRegionIterArgs()[en.index()]);
      oldOp->replaceAllUsesWith(arg);
    }
    rewriter.replaceOp(forOp, repResults);
    return success();
  }
};

struct ForOpCanonicalizationPass
    : PassWrapper<ForOpCanonicalizationPass, FunctionPass> {
  void runOnFunction() override {
    FuncOp fn = getFunction();
    OwningRewritePatternList patterns;
    patterns.insert<ForOpArgFolding>(fn.getContext());
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<FunctionPass> createForOpCanonicalizationPass() {
  return std::make_unique<ForOpCanonicalizationPass>();
}

static PassRegistration<ForOpCanonicalizationPass> pass(
    "iree-codegen-canonicalize-scf-for",
    "An ad-hoc pass to canonicalize selected loop carried dependencies on "
    "scf.for",
    [] { return std::make_unique<ForOpCanonicalizationPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
