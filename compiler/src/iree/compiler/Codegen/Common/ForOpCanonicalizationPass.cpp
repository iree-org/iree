// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOROPCANONICALIZATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct CanonicalizeForOpInductionVarShape final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // Return true if it might be possible to yield the operand of `op` instead of
  // its result.
  bool canFoldEnd(Operation *op) const {
    if (!isa<vector::ShapeCastOp, vector::BroadcastOp,
             UnrealizedConversionCastOp>(op)) {
      return false;
    }
    if (op->getNumOperands() != 1) {
      return false;
    }
    return true;
  }

  // Return true if `op` (user of induction variable) can be folded if the
  // induction variable is changed to have type `type`.
  bool canFoldStart(Operation *op, Type type) const {
    if (op->getNumResults() != 1) {
      return false;
    }
    if (!isa<vector::ShapeCastOp, vector::ExtractOp,
             UnrealizedConversionCastOp>(op)) {
      return false;
    }
    if (op->getResult(0).getType() != type) {
      return false;
    }
    return true;
  }

  // Transfer the body of `source` into `dest` and update the terminator of
  // `dest` to use the specified results. The result list also replaces
  // any block arguments from `source` with the corresponding block argument
  // in `dest` and returns the updated result list.
  SmallVector<Value> transferBody(Block *source, Block *dest,
                                  ArrayRef<Value> results,
                                  PatternRewriter &rewriter) const {
    // Collect the old block arguments before merging.
    SmallVector<std::optional<int64_t>> maybeBlockArgNum;
    for (auto res : results) {
      if (auto blockArg = dyn_cast<BlockArgument>(res)) {
        maybeBlockArgNum.push_back(blockArg.getArgNumber());
      } else {
        maybeBlockArgNum.push_back(std::nullopt);
      }
    }
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    // Create a new result set with the updated block arguments.
    SmallVector<Value> newResults;
    for (auto [index, argNum] : llvm::enumerate(maybeBlockArgNum)) {
      if (argNum) {
        newResults.push_back(dest->getArgument(*argNum));
      } else {
        newResults.push_back(results[index]);
      }
    }
    rewriter.modifyOpInPlace(
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
    for (unsigned index = 0, s = forOp.getNumRegionIterArgs(); index < s;
         ++index) {

      Operation *returnValDef = returnValues[index].getDefiningOp();

      // Currently we don't track usage through block arguments.
      if (!returnValDef) {
        continue;
      }

      if (!canFoldEnd(returnValDef)) {
        continue;
      }
      Value newReturn = returnValDef->getOperand(0);

      BlockArgument iterArg = forOp.getRegionIterArg(index);

      // Check that all of the users of the induction variable can be folded,
      // and if so return the final user of the induction variable.
      Operation *finalIvUser = [&]() -> Operation * {
        Operation *op = nullptr;
        for (OpOperand &use : iterArg.getUses()) {
          op = use.getOwner();
          if (!canFoldStart(op, newReturn.getType())) {
            return nullptr;
          }
        }
        return op;
      }();

      if (!finalIvUser) {
        continue;
      }

      iteratorFolded.push_back(index);
      resultOps.push_back(returnValDef);
      returnValues[index] = newReturn;

      // Create a clone of the user of the induction variable. Clone will be
      // used as induction variable in new scf.for.
      //
      // An example, where %new_clone is clone of %to_clone:
      // ```
      // %0 = scf.for %arg0 = %c1 to %c3 step %c1 iter_args(%arg1 = %cst) ->
      // (vector<4xf32>) {
      //   %to_clone = vector.extract %arg1[%arg0] : f32 from vector<4xf32>
      //   ...
      // }
      // %new_clone = vector.extact %cst[%c1] : f32 from vector<4xf32>
      // ```
      IRMapping mapping;
      mapping.map(iterArg, initArgs[index]);

      std::optional<SmallVector<Value>> loopIndVars =
          forOp.getLoopInductionVars();
      assert(loopIndVars.has_value() && loopIndVars.value().size() == 1 &&
             "scf.for must have 1 loop induction variable");
      Value loopIndVar = loopIndVars.value()[0];
      Value start = forOp.getLowerBound();
      mapping.map(loopIndVar, start);
      initArgs[index] = rewriter.clone(*finalIvUser, mapping)->getResult(0);
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
      repResults[iter] =
          rewriter.clone(*resultOps[index], mapping)->getResult(0);

      for (OpOperand &use : newLoop.getRegionIterArgs()[iter].getUses()) {
        Operation *oldOp = use.getOwner();
        assert(oldOp->getNumResults() == 1 && "expected single result");
        rewriter.replaceAllUsesWith(oldOp->getResult(0),
                                    newLoop.getRegionIterArgs()[iter]);
      }
    }
    rewriter.replaceOp(forOp, repResults);

    return success();
  }
};

/// An ad-hoc pattern to convert scf.for loop-carried values of < 128 total
/// bits, but more than 4 elements. For example, insert `vector.bitcast` ops to
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
      int64_t numElements = ShapedType::getNumElements(iterType.getShape());
      int64_t bitWidth = IREE::Util::getTypeBitWidth(iterType.getElementType());
      int64_t totalBits = numElements * bitWidth;
      if (numElements > 4 && totalBits <= 128 &&
          llvm::isPowerOf2_64(totalBits)) {
        ivIndices.push_back(index);
        ivTypes.push_back(iterType);
        auto shapeCastType =
            VectorType::get({numElements}, iterType.getElementType());
        castTypes.push_back(shapeCastType);
        auto targetType =
            VectorType::get({llvm::divideCeilSigned(totalBits, 32)},
                            rewriter.getIntegerType(
                                std::min(static_cast<int64_t>(32), totalBits)));
        targetTypes.push_back(targetType);
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

struct ForOpCanonicalizationPass final
    : impl::ForOpCanonicalizationPassBase<ForOpCanonicalizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // These patterns collide so we apply them one after another. The
    // canonicalization pattern will be blocked by the packing pattern
    // so we apply that first.
    RewritePatternSet canonPatterns(&getContext());
    canonPatterns.add<CanonicalizeForOpInductionVarShape>(fn.getContext());
    if (failed(applyPatternsGreedily(fn, std::move(canonPatterns)))) {
      return signalPassFailure();
    }
    RewritePatternSet packPatterns(&getContext());
    packPatterns.add<PackForOpInductionVarVector>(fn.getContext());
    if (failed(applyPatternsGreedily(fn, std::move(packPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateForOpInductionVarShapePatterns(RewritePatternSet &patterns,
                                            PatternBenefit benefit) {
  patterns.add<CanonicalizeForOpInductionVarShape>(patterns.getContext(),
                                                   benefit);
}
} // namespace mlir::iree_compiler
