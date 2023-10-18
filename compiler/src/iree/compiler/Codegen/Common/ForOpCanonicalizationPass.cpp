// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pattern to combine instructions across ForOp boundary. It is common when
/// doing incremental lowering to generate transient ops that cancel each others
/// out. Canonicalization usually clean up those operations. When the value is
/// loop carried, MLIR canonicalization currently doesn't remove the redundant
/// operations.
///
/// This pass allow to workaround MLIR limitation and does ad hoc clean up of
/// instructions found in IREE. Once we have a more general mechanism in MLIR
/// this pass can be completely removed.
/// This pass does this kind of transformation:
/// ```
/// %21 = vector.shape_cast %20 : vector<4xf32> to vector<1x4xf32>
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %21)
///    -> vector<1x4xf32> {
///    [...]
///    %100 = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
///    [...]
///    %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
///    scf.yield %109 : vector<1x4xf32>
///  }
///  %24 = vector.shape_cast %22 : vector<1x4xf32> to vector<4xf32>
/// ```
/// ->
/// ```
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %20)
///    -> vector<4xf32> {
///    [...]
///    scf.yield %108 : vector<4xf32>
///  }
/// ```
struct CanonicalizeForOpInductionVarShape final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  Value FoldCarryDep(scf::ForOp forOp, Operation *ivUser,
                     Operation *ivDef) const {
    if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(ivUser)) {
      if (auto souceOp = dyn_cast<vector::ShapeCastOp>(ivDef)) {
        if (shapeCast.getType() == souceOp.getSource().getType()) {
          return souceOp.getSource();
        }
      }
    } else if (auto extractOp = dyn_cast<vector::ExtractOp>(ivUser)) {
      if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(ivDef)) {
        if (extractOp.getType() == broadcastOp.getSourceType()) {
          return broadcastOp.getSource();
        }
      }
    } else if (auto targetOp = dyn_cast<UnrealizedConversionCastOp>(ivUser)) {
      if (auto sourceOp = dyn_cast<UnrealizedConversionCastOp>(ivDef)) {
        if (sourceOp->getNumOperands() == 1 && targetOp->getNumResults() == 1 &&
            sourceOp->getOperandTypes().front() ==
                targetOp.getResultTypes().front()) {
          return sourceOp.getInputs().front();
        }
      }
    }
    return Value();
  }

  void transferBody(Block *source, Block *dest, ArrayRef<Value> results,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    rewriter.updateRootInPlace(
        yieldOp, [&]() { yieldOp.getOperation()->setOperands(results); });
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned, 8> iteratorFolded;
    SmallVector<Operation *, 8> resultOps;
    auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto returnValues = llvm::to_vector<8>(terminator.getOperands());
    auto initArgs = llvm::to_vector<8>(forOp.getInitArgs());
    for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!iterArg.hasOneUse())
        continue;
      Operation *op = iterArg.use_begin()->getOwner();
      if (!isa<vector::ShapeCastOp, vector::ExtractOp,
               UnrealizedConversionCastOp>(op)) {
        continue;
      }
      Operation *returnValDef = returnValues[index].getDefiningOp();
      Value newReturn = FoldCarryDep(forOp, op, returnValDef);
      if (!newReturn)
        continue;
      iteratorFolded.push_back(index);
      resultOps.push_back(returnValDef);
      returnValues[index] = newReturn;

      IRMapping mapping;
      mapping.map(iterArg, initArgs[index]);
      initArgs[index] = rewriter.clone(*op, mapping)->getResult(0);
    }
    if (iteratorFolded.empty())
      return failure();
    auto newLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs);
    transferBody(forOp.getBody(), newLoop.getBody(), returnValues, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 8> repResults(newLoop.getResults().begin(),
                                     newLoop.getResults().end());
    for (auto [index, iter] : llvm::enumerate(iteratorFolded)) {
      IRMapping mapping;
      mapping.map(returnValues[iter], newLoop.getResult(iter));
      repResults[index] =
          rewriter.clone(*resultOps[index], mapping)->getResult(0);
      Operation *oldOp =
          newLoop.getRegionIterArgs()[index].use_begin()->getOwner();
      assert(oldOp->getNumResults() == 1 && "expected single result");
      rewriter.replaceAllUsesWith(oldOp->getResult(0),
                                  newLoop.getRegionIterArgs()[index]);
    }
    rewriter.replaceOp(forOp, repResults);
    return success();
  }
};

/// An ad-hoc pattern to convert scf.for loop-carried values from
/// `vector<8xf16>` to `vector<4xf32>` by inserting `vector.bitcast` around
/// scf.for boundaries.
///
/// Those loop-carried values will be lowered into SPIR-V local variables. This
/// pattern allows packing f16 values into f32 variables tightly so that we can
/// generate shader conformant SPIR-V.
struct PackForOpInductionVarVector final : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    VectorType v8f16Type = VectorType::get({8}, rewriter.getF16Type());
    VectorType v4f32Type = VectorType::get({4}, rewriter.getF32Type());

    SmallVector<unsigned, 8> ivIndices;
    for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (iterArg.getType() == v8f16Type)
        ivIndices.push_back(index);
    }
    if (ivIndices.empty())
      return failure();

    // Bit cast all init values from v8f16 to v4f32.
    auto ivInitValues = llvm::to_vector<8>(forOp.getInitArgs());
    for (unsigned index : ivIndices) {
      Value oldValue = ivInitValues[index];
      ivInitValues[index] = rewriter.create<vector::BitCastOp>(
          oldValue.getLoc(), v4f32Type, oldValue);
    }

    // Create a new loop with the casted init values. This also creates
    // induction variables with proper type.
    auto newLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), ivInitValues);

    // Move all operations to the new for op. This also replaces block
    // arguments. to the new block arguments.
    rewriter.mergeBlocks(forOp.getBody(), newLoop.getBody(),
                         newLoop.getBody()->getArguments());

    // Bit cast induction variables back to the original type to fix uses.
    rewriter.setInsertionPointToStart(newLoop.getBody());
    for (unsigned index : ivIndices) {
      Value newIv = newLoop.getRegionIterArgs()[index];
      auto bitcastOp =
          rewriter.create<vector::BitCastOp>(newIv.getLoc(), v8f16Type, newIv);
      // Replace all uses of the new induction variable with a bitcast. We need
      // to exclude the bitcast op itself given it also uses the induction
      // variable.
      SmallPtrSet<Operation *, 1> exceptions{bitcastOp};
      newIv.replaceAllUsesExcept(bitcastOp, exceptions);
    }

    auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    auto ivRetValues = llvm::to_vector<8>(yieldOp.getOperands());

    // Bit cast return values to the new type to fix yield.
    rewriter.setInsertionPoint(yieldOp);
    for (unsigned index : ivIndices) {
      Value oldRet = ivRetValues[index];
      ivRetValues[index] = rewriter.create<vector::BitCastOp>(
          oldRet.getLoc(), v4f32Type, oldRet);
    }
    yieldOp->setOperands(ivRetValues);

    SmallVector<Value, 8> forRetValues;
    for (Value result : newLoop.getResults())
      forRetValues.push_back(result);

    // Bit cast return values to the old type to fix for op uses.
    rewriter.setInsertionPointAfter(newLoop);
    for (unsigned index : ivIndices) {
      Value oldRet = forRetValues[index];
      forRetValues[index] = rewriter.create<vector::BitCastOp>(
          oldRet.getLoc(), v8f16Type, oldRet);
    }

    rewriter.replaceOp(forOp, forRetValues);
    return success();
  }
};

struct ForOpCanonicalizationPass
    : public ForOpCanonicalizationBase<ForOpCanonicalizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.insert<CanonicalizeForOpInductionVarShape,
                    PackForOpInductionVarVector>(fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createForOpCanonicalizationPass() {
  return std::make_unique<ForOpCanonicalizationPass>();
}

} // namespace iree_compiler
} // namespace mlir
