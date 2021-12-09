// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Folds a value that has its definition and usage across SCF region
/// boundaries if possible. Returns null value otherwise.
///
/// Such cases won't have def-use relationship via direct SSA reference; rather
/// it's via value yielding and usage of the SCF op as a whole.
static Value foldValueDefUse(Operation* user, Operation* def) {
  if (auto shapeCast = dyn_cast_or_null<vector::ShapeCastOp>(user)) {
    if (auto souceOp = dyn_cast_or_null<vector::ShapeCastOp>(def)) {
      if (shapeCast.getType() == souceOp.source().getType())
        return souceOp.source();
    }
  } else if (auto extractOp = dyn_cast_or_null<vector::ExtractOp>(user)) {
    if (auto broadcastOp = dyn_cast_or_null<vector::BroadcastOp>(def)) {
      if (extractOp.getType() == broadcastOp.getSourceType())
        return broadcastOp.source();
      // If we are extracting a scalar element, it's also okay to fold.
      if (extractOp.getType().isIntOrFloat()) return broadcastOp.source();
    }
  } else if (auto targetOp =
                 dyn_cast_or_null<UnrealizedConversionCastOp>(user)) {
    if (auto sourceOp = dyn_cast_or_null<UnrealizedConversionCastOp>(def)) {
      if (sourceOp->getNumOperands() == 1 && targetOp->getNumResults() == 1 &&
          sourceOp->getOperandTypes().front() ==
              targetOp.getResultTypes().front())
        return sourceOp.inputs().front();
    }
  }
  return Value();
}

/// Moves all ops from the `source` block into the `dest` block and updates the
/// yield terminator op to yield values in `results`.
static void transferBody(Block* source, Block* dest, ArrayRef<Value> results,
                         RewriterBase& rewriter) {
  // Move all operations to the destination block.
  rewriter.mergeBlocks(source, dest, dest->getArguments());
  // Replace the yield op by one that returns only the used values.
  auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
  yieldOp.getOperation()->setOperands(results);
}

/// Fold reciprocal ops across ForOp boundary. It is common when doing
/// incremental lowering to generate transient ops that cancel each others out.
/// Canonicalization usually clean up those operations. When the value is loop
/// carried, MLIR canonicalization currently doesn't remove the redundant
/// operations.
///
/// This pass allow to workaround MLIR limitation and does ad hoc clean up of
/// instructions found in IREE. Once we have a more general mechanism in MLIR
/// this pass can be completely removed.
///
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
struct CanoncalizeForOpCarriedValueShape final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<unsigned, 8> foldedCarriedValueIndices;
    SmallVector<Operation*, 8> resultOps;

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto yieldValues = llvm::to_vector<8>(yieldOp.getOperands());
    auto initValues = llvm::to_vector<8>(forOp.getIterOperands());

    for (const auto& it : llvm::enumerate(forOp.getRegionIterArgs())) {
      unsigned index = it.index();
      Value iterArg = it.value();
      if (!iterArg.hasOneUse()) continue;

      Operation* userOp = iterArg.use_begin()->getOwner();
      if (!isa<vector::ShapeCastOp, vector::ExtractOp,
               UnrealizedConversionCastOp>(userOp))
        continue;

      Operation* defOp = yieldValues[index].getDefiningOp();
      Value newYieldValue = foldValueDefUse(userOp, defOp);
      if (!newYieldValue) continue;

      foldedCarriedValueIndices.push_back(index);
      resultOps.push_back(defOp);
      yieldValues[index] = newYieldValue;

      // The current iterator argument involves shape ops that can be cancelled
      // across the region boundary. This means pushing the original user op to
      // before the region. So clone it outside and take the initial value as
      // operands.
      BlockAndValueMapping mapping;
      mapping.map(iterArg, initValues[index]);
      initValues[index] = rewriter.clone(*userOp, mapping)->getResult(0);
    }
    if (foldedCarriedValueIndices.empty()) return failure();

    auto newLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.lowerBound(), forOp.upperBound(), forOp.step(),
        initValues);
    transferBody(forOp.getBody(), newLoop.getBody(), yieldValues, rewriter);

    // Replace the results with the new loop's results.
    SmallVector<Value, 8> newResults(newLoop.getResults().begin(),
                                     newLoop.getResults().end());
    for (const auto& it : llvm::enumerate(foldedCarriedValueIndices)) {
      // The current iterator argument involves shape ops that can be cancelled
      // across the region boundary. This means pushing the original def to
      // after the region. So clone it outside and take the new result as
      // operands.
      BlockAndValueMapping mapping;
      mapping.map(yieldValues[it.value()], newLoop.getResult(it.value()));
      newResults[it.index()] =
          rewriter.clone(*resultOps[it.index()], mapping)->getResult(0);

      Operation* oldOp =
          newLoop.getRegionIterArgs()[it.index()].use_begin()->getOwner();
      SmallVector<Value, 1> arg(1, newLoop.getRegionIterArgs()[it.index()]);
      oldOp->replaceAllUsesWith(arg);
    }
    rewriter.replaceOp(forOp, newResults);
    return success();
  }
};

/// Fold reciprocal ops across IfOp boundary. This is similar to the above
/// pattern for ForOps.
///
/// This pass does this kind of transformation:
/// ```
/// %0 = scf.if %cond -> (vector<1x4xf32>) {
///   %1 = scf.broadcast %scalar : f32 into vector<1x4xf32>
/// } else {
///   scf.yield %zero : vector<1x4xf32>
/// }
/// %2 = vector.extract %0[0, 0] : vector<1x4xf32>
/// ```
/// ->
/// ```
/// %0 = scf.if %cond -> (f32) {
///   scf.yield %scalar : f32
/// } else {
///   scf.yield %zero : f32
/// }
/// ```
struct CanoncalizeIfReturnedValueShape final
    : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter& rewriter) const override {
    auto resultTypes = llvm::to_vector<8>(ifOp.getResultTypes());
    if (resultTypes.empty()) return failure();

    // Require all yield values in the "else" branch to be splat constant for
    // now. Merging different cases between the "then" and "else" branches can
    // be complicated; leave that to only when needed.
    if (ifOp.elseBlock()) {
      SplatElementsAttr splat;
      for (Value v : ifOp.elseYield().getOperands()) {
        if (!matchPattern(v, m_Constant(&splat))) return failure();
      }
    }

    // All yielded values from the "then" branch.
    auto thenYieldValues = llvm::to_vector<8>(ifOp.thenYield().getOperands());
    // Shape casting ops that should be used to build the new results.
    SmallVector<std::pair<unsigned, Operation*>, 8> resultOps;

    bool folded = false;
    for (const auto& it : llvm::enumerate(thenYieldValues)) {
      unsigned index = it.index();
      Value result = ifOp.getResult(index);
      if (!result.hasOneUse()) continue;
      Operation* userOp = result.use_begin()->getOwner();

      // Only support broadcast-extract pairs for now.
      if (!isa<vector::ExtractOp>(userOp)) continue;

      Operation* defOp = it.value().getDefiningOp();
      Value newYieldValue = foldValueDefUse(userOp, defOp);
      if (!newYieldValue) continue;

      resultTypes[index] = newYieldValue.getType();
      // Update to record the source value to yield after pushing shape casting
      // ops to be after the if region.
      thenYieldValues[index] = newYieldValue;
      // Also record the shape cast op so later we can clone it after the
      // if region.
      resultOps.emplace_back(index, defOp);
      folded = true;
    }

    if (!folded) return failure();

    auto newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes,
                                            ifOp.condition(), ifOp.elseBlock());

    transferBody(ifOp.thenBlock(), newIf.thenBlock(), thenYieldValues,
                 rewriter);

    if (ifOp.elseBlock()) {
      OpBuilder elseBuilder = newIf.getElseBodyBuilder(rewriter.getListener());
      auto elseYieldValues = llvm::to_vector<8>(ifOp.elseYield().getOperands());
      // Reshape all splat constants to match shape.
      for (int i = 0; i < elseYieldValues.size(); ++i) {
        Value oldValue = elseYieldValues[i];
        auto oldAttr = oldValue.getDefiningOp<arith::ConstantOp>().getValue();
        Attribute newAttr;
        if (auto shapedType = resultTypes[i].dyn_cast<ShapedType>()) {
          newAttr = oldAttr.cast<SplatElementsAttr>().reshape(shapedType);
        } else {
          newAttr =
              oldAttr.cast<SplatElementsAttr>().getSplatValue<Attribute>();
        }
        elseYieldValues[i] =
            elseBuilder.create<arith::ConstantOp>(oldValue.getLoc(), newAttr);
      }
      elseBuilder.create<scf::YieldOp>(ifOp.elseYield().getLoc(),
                                       elseYieldValues);
    }

    // Replace the results with the new if's results.
    SmallVector<Value, 8> newResults(newIf.getResults().begin(),
                                     newIf.getResults().end());
    for (const auto& it : resultOps) {
      // Push the original def to after the region. So clone it outside and take
      // the new result as operands.
      BlockAndValueMapping mapping;
      mapping.map(thenYieldValues[it.first], newIf.getResult(it.first));
      newResults[it.first] = rewriter.clone(*it.second, mapping)->getResult(0);
    }

    rewriter.replaceOp(ifOp, newResults);
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
                                PatternRewriter& rewriter) const override {
    VectorType v8f16Type = VectorType::get({8}, rewriter.getF16Type());
    VectorType v4f32Type = VectorType::get({4}, rewriter.getF32Type());

    SmallVector<unsigned, 8> ivIndices;
    for (const auto& it : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (it.value().getType() == v8f16Type) ivIndices.push_back(it.index());
    }
    if (ivIndices.empty()) return failure();

    // Bit cast all init values from v8f16 to v4f32.
    auto ivInitValues = llvm::to_vector<8>(forOp.getIterOperands());
    for (unsigned index : ivIndices) {
      Value oldValue = ivInitValues[index];
      ivInitValues[index] = rewriter.create<vector::BitCastOp>(
          oldValue.getLoc(), v4f32Type, oldValue);
    }

    // Create a new loop with the casted init values. This also creates
    // induction variables with proper type.
    auto newLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.lowerBound(), forOp.upperBound(), forOp.step(),
        ivInitValues);

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
      SmallPtrSet<Operation*, 1> exceptions{bitcastOp};
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
    for (Value result : newLoop.getResults()) forRetValues.push_back(result);

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
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    FuncOp fn = getOperation();
    MLIRContext* context = &getContext();
    OwningRewritePatternList patterns(context);
    patterns
        .insert<CanoncalizeForOpCarriedValueShape,
                CanoncalizeIfReturnedValueShape, PackForOpInductionVarVector>(
            context);
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createForOpCanonicalizationPass() {
  return std::make_unique<ForOpCanonicalizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
