// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

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

  SmallVector<Value> transferBody(Block *source, Block *dest,
                                  ArrayRef<Value> results,
                                  PatternRewriter &rewriter) const {
    // Collect the old block arguments before merging.
    SmallVector<int64_t> maybeBlockArgNum;
    for (auto res : results) {
      if (auto blockArg = dyn_cast<BlockArgument>(res)) {
        maybeBlockArgNum.push_back(blockArg.getArgNumber());
      } else {
        maybeBlockArgNum.push_back(-1);
      }
    }
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    // Create a new result set with the updated block arguments.
    SmallVector<Value> newResults;
    for (auto [index, argNum] : llvm::enumerate(maybeBlockArgNum)) {
      if (argNum >= 0) {
        newResults.push_back(dest->getArgument(argNum));
      } else {
        newResults.push_back(results[index]);
      }
    }
    rewriter.updateRootInPlace(
        yieldOp, [&]() { yieldOp.getOperation()->setOperands(newResults); });
    return newResults;
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
      // Currently we don't track usage through block arguments.
      if (!returnValDef) {
        continue;
      }
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
    SmallVector<Value> newReturnVals = transferBody(
        forOp.getBody(), newLoop.getBody(), returnValues, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 8> repResults(newLoop.getResults().begin(),
                                     newLoop.getResults().end());
    for (auto [index, iter] : llvm::enumerate(iteratorFolded)) {
      IRMapping mapping;
      mapping.map(newReturnVals[iter], newLoop.getResult(iter));
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

/// An ad-hoc pattern to convert scf.for loop-carried values of < 128 total
/// bits, but more than 4 elements. For example, insert `vector.bitcasts` to
/// cast `vector<8xf16>` to `vector<4xf32>`. To handle `vector<2x4xf16>` and
/// `vector<1x8xf16>`, shape casts are inserted to before the bitcast to align
/// the ranks
///
/// Those loop-carried values will be lowered into SPIR-V local variables. This
/// pattern allows packing i4/i8/f16 values into i32 variables tightly so that
/// we can generate shader conformant SPIR-V.
struct PackForOpInductionVarVector final : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned, 8> ivIndices;
    SmallVector<VectorType, 8> ivTypes;
    SmallVector<VectorType, 8> castTypes;
    SmallVector<VectorType, 8> targetTypes;
    for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      VectorType iterType = dyn_cast<VectorType>(iterArg.getType());
      if (!iterType || iterType.getRank() == 0) {
        continue;
      }
      auto shape = iterType.getShape();
      auto numElements = std::accumulate(shape.begin(), shape.end(), 1,
                                         std::multiplies<int64_t>());
      int64_t bitWidth = iterType.getElementType().getIntOrFloatBitWidth();
      int64_t totalBits = numElements * bitWidth;
      if (numElements > 4 && totalBits <= 128 &&
          llvm::isPowerOf2_64(totalBits)) {
        ivIndices.push_back(index);
        ivTypes.push_back(iterType);
        castTypes.push_back(
            VectorType::get({numElements}, iterType.getElementType()));
        targetTypes.push_back(
            VectorType::get({totalBits / 32 + (totalBits % 32 != 0)},
                            rewriter.getIntegerType(std::min(
                                static_cast<int64_t>(32), totalBits))));
      }
    }
    if (ivIndices.empty())
      return failure();

    // Bit cast all init values to the smaller vector (fewer elements).
    auto ivInitValues = llvm::to_vector<8>(forOp.getInitArgs());
    for (auto [index, castType, targetType] :
         llvm::zip_equal(ivIndices, castTypes, targetTypes)) {
      Value oldValue = ivInitValues[index];
      Value shapeCast = rewriter.create<vector::ShapeCastOp>(
          oldValue.getLoc(), castType, oldValue);
      ivInitValues[index] = rewriter.create<vector::BitCastOp>(
          oldValue.getLoc(), targetType, shapeCast);
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
    for (auto [index, castType, origType] :
         llvm::zip_equal(ivIndices, castTypes, ivTypes)) {
      Value newIv = newLoop.getRegionIterArgs()[index];
      auto bitcastOp =
          rewriter.create<vector::BitCastOp>(newIv.getLoc(), castType, newIv);
      auto shapeCastOp = rewriter.create<vector::ShapeCastOp>(
          newIv.getLoc(), origType, bitcastOp);
      // Replace all uses of the new induction variable with a bitcast. We need
      // to exclude the bitcast op itself given it also uses the induction
      // variable.
      SmallPtrSet<Operation *, 2> exceptions{bitcastOp};
      newIv.replaceAllUsesExcept(shapeCastOp, exceptions);
    }

    auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    auto ivRetValues = llvm::to_vector<8>(yieldOp.getOperands());

    // Bit cast return values to the new type to fix yield.
    rewriter.setInsertionPoint(yieldOp);
    for (auto [index, castType, targetType] :
         llvm::zip_equal(ivIndices, castTypes, targetTypes)) {
      Value oldRet = ivRetValues[index];
      Value shapeCast = rewriter.create<vector::ShapeCastOp>(oldRet.getLoc(),
                                                             castType, oldRet);
      ivRetValues[index] = rewriter.create<vector::BitCastOp>(
          oldRet.getLoc(), targetType, shapeCast);
    }
    yieldOp->setOperands(ivRetValues);

    SmallVector<Value, 8> forRetValues;
    for (Value result : newLoop.getResults())
      forRetValues.push_back(result);

    // Bit cast return values to the old type to fix for op uses.
    rewriter.setInsertionPointAfter(newLoop);
    for (auto [index, castType, origType] :
         llvm::zip_equal(ivIndices, castTypes, ivTypes)) {
      Value oldRet = forRetValues[index];
      Value bitCast =
          rewriter.create<vector::BitCastOp>(oldRet.getLoc(), castType, oldRet);
      forRetValues[index] = rewriter.create<vector::ShapeCastOp>(
          oldRet.getLoc(), origType, bitCast);
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
    // These patterns collide so we apply them one after another. The
    // canonicalization pattern will be blocked by the packing pattern
    // so we apply that first.
    RewritePatternSet canonPatterns(&getContext());
    canonPatterns.insert<CanonicalizeForOpInductionVarShape>(fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(canonPatterns)))) {
      return signalPassFailure();
    }
    RewritePatternSet packPatterns(&getContext());
    packPatterns.insert<PackForOpInductionVarVector>(fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(packPatterns)))) {
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
