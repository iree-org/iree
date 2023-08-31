// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LLVMCPUFoldVectorContractUnitDims.cpp - Pass to fold unit dims of
// vector.contract ops -===//
//
// Patterns to fold away unit dimensions on `vector.contract` ops
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-fold-unit-reduction-dims"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {

// Given a `vector.contract` op and a set of indices to fold, this op rewrites
// the `vector.contract` op with surrounding `vector.shape_cast` ops to fold
// away the indicated indices.
static FailureOr<Value>
dropFoldableUnitIndices(PatternRewriter &rewriter,
                        vector::ContractionOp contractOp,
                        SmallVector<int64_t> foldIndices) {
  SmallVector<int64_t> contractShape = *contractOp.getShapeForUnroll();
  SmallVector<vector::IteratorType> iteratorTypes =
      contractOp.getIteratorTypesArray();
  auto indexingMaps = contractOp.getIndexingMapsArray();
  SmallVector<SmallVector<int64_t>> dstShapes;
  SmallVector<SmallVector<AffineExpr>> dstExprs;
  SmallVector<Value> inputs(
      {contractOp.getLhs(), contractOp.getRhs(), contractOp.getAcc()});
  llvm::SetVector<int64_t> foldableDims;
  for (int64_t dim : foldIndices)
    foldableDims.insert(dim);

  for (AffineMap map : indexingMaps) {
    SmallVector<int64_t> dstShape;
    SmallVector<AffineExpr> dstExpr;
    for (const auto &expr : enumerate(map.getResults())) {
      if (auto dimExpr = expr.value().dyn_cast<AffineDimExpr>()) {
        if (!foldableDims.contains(dimExpr.getPosition())) {
          dstShape.push_back(contractShape[dimExpr.getPosition()]);
          unsigned numSkipped = 0;
          for (int64_t ind : foldIndices) {
            if (dimExpr.getPosition() > ind) {
              numSkipped++;
            }
          }
          dstExpr.push_back(
              rewriter.getAffineDimExpr(dimExpr.getPosition() - numSkipped));
        }
      } else {
        return failure();
      }
    }
    dstShapes.push_back(dstShape);
    dstExprs.push_back(dstExpr);
  }

  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<vector::IteratorType> newIteratorTypes;
  for (auto iter : enumerate(iteratorTypes)) {
    if (!foldableDims.contains(iter.index())) {
      newIteratorTypes.push_back(iter.value());
    }
  }

  for (int i = 0; i < 3; i++) {
    // Shape unchanged
    if (dstShapes[i].size() == indexingMaps[i].getResults().size()) {
      newInputs.push_back(inputs[i]);
      AffineMap newIndexingMap =
          AffineMap::get(/*dimCount=*/contractShape.size() - foldIndices.size(),
                         /*symCount=*/0, dstExprs[i], contractOp.getContext());
      newIndexingMaps.push_back(newIndexingMap);
      continue;
    }
    if (dstShapes[i].size() == 0) {
      return failure();
    }
    VectorType inputVecType = llvm::cast<VectorType>(inputs[i].getType());
    VectorType dstType =
        VectorType::get(dstShapes[i], inputVecType.getElementType());

    Value result;
    auto extsiop = inputs[i].getDefiningOp<arith::ExtSIOp>();
    auto extuiop = inputs[i].getDefiningOp<arith::ExtUIOp>();
    if (!extsiop && !extuiop) {
      result = rewriter.create<vector::ShapeCastOp>(contractOp.getLoc(),
                                                    dstType, inputs[i]);
    } else {
      Value extIn = extsiop ? extsiop.getIn() : extuiop.getIn();
      VectorType extInType = llvm::dyn_cast<VectorType>(extIn.getType());
      VectorType shapeCastOutType =
          VectorType::get(dstType.getShape(), extInType.getElementType());
      Value shapeCastResult = rewriter.create<vector::ShapeCastOp>(
          contractOp.getLoc(), shapeCastOutType, extIn);
      result = extsiop ? rewriter
                             .create<arith::ExtSIOp>(contractOp.getLoc(),
                                                     dstType, shapeCastResult)
                             .getResult()
                       : rewriter
                             .create<arith::ExtUIOp>(contractOp.getLoc(),
                                                     dstType, shapeCastResult)
                             .getResult();
    }
    AffineMap newIndexingMap =
        AffineMap::get(/*dimCount=*/contractShape.size() - foldIndices.size(),
                       /*symCount=*/0, dstExprs[i], contractOp.getContext());
    newInputs.push_back(result);
    newIndexingMaps.push_back(newIndexingMap);
  }
  auto newContract =
      rewriter
          .create<vector::ContractionOp>(
              contractOp.getLoc(), newInputs[0], newInputs[1], newInputs[2],
              rewriter.getAffineMapArrayAttr(newIndexingMaps),
              rewriter.getArrayAttr(llvm::to_vector(llvm::map_range(
                  newIteratorTypes,
                  [&](vector::IteratorType t) -> mlir::Attribute {
                    return vector::IteratorTypeAttr::get(rewriter.getContext(),
                                                         t);
                  }))))
          .getResult();
  return newContract;
}

// This pattern matches on a `vector.contract` op with unit size dimensions, and
// folds these dimensions away
class DropVectorContractUnitDims final
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    LDBG("vector.contract op:\n" << contractOp);
    VectorType outputType =
        llvm::dyn_cast<VectorType>(contractOp.getAcc().getType());
    if (!outputType) {
      return failure();
    }

    auto iteratorTypes = contractOp.getIteratorTypesArray();
    SmallVector<int64_t> contractDims = *contractOp.getShapeForUnroll();
    unsigned numParallel = 0;
    unsigned numReduction = 0;
    SmallVector<int64_t> unitParallelDims;
    SmallVector<int64_t> unitReductionDims;
    SmallVector<int64_t> foldableDims;
    for (auto size : enumerate(contractDims)) {
      if (iteratorTypes[size.index()] == vector::IteratorType::parallel) {
        numParallel++;
        if (size.value() == 1) {
          unitParallelDims.push_back(size.index());
        }
      } else {
        numReduction++;
        if (size.value() == 1) {
          unitReductionDims.push_back(size.index());
        }
      }
    }
    if (numReduction && numReduction == unitReductionDims.size()) {
      foldableDims.append(unitReductionDims.begin(),
                          unitReductionDims.end() - 1);
    } else {
      foldableDims.append(unitReductionDims.begin(), unitReductionDims.end());
    }
    if (numParallel && numParallel == unitParallelDims.size()) {
      foldableDims.append(unitParallelDims.begin() + 1, unitParallelDims.end());
    } else {
      foldableDims.append(unitParallelDims.begin(), unitParallelDims.end());
    }
    if (!foldableDims.size()) {
      return failure();
    }

    FailureOr<Value> maybeNewContract =
        dropFoldableUnitIndices(rewriter, contractOp, foldableDims);
    if (failed(maybeNewContract)) {
      return failure();
    }
    Value newContract = maybeNewContract.value();
    LDBG("Replaced vector.contract:\n" << newContract);

    VectorType newOutputType =
        llvm::dyn_cast<VectorType>(newContract.getType());
    if (outputType != newOutputType) {
      // Reshape output of new vector.contract if needed
      Value shapeCastResult = rewriter.create<vector::ShapeCastOp>(
          contractOp.getLoc(), outputType, newContract);
      rewriter.replaceOp(contractOp, shapeCastResult);
    } else {
      rewriter.replaceOp(contractOp, newContract);
    }

    return success();
  }
};

// This pattern matches on a sequence of
// `vector.shape_cast->vector.contract->vector.shape_cast` within an `scf.for`
// op, where the shape cast ops are casting an argument of the `scf.for` op and
// the yielded result of the `scf.for` op. Once matched, the `vector.shape_cast`
// ops are hoisted out of the `scf.for` op.
class HoistShapeCastOutOfSCFFor final : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    LDBG("forOp:\n" << forOp);
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    std::optional<std::pair<vector::ShapeCastOp, vector::ShapeCastOp>>
        hoistableShapeCast = std::nullopt;
    int initArgIdx;
    for (Value result : yieldOp.getOperation()->getOperands()) {
      auto outputShapeCastOp = result.getDefiningOp<vector::ShapeCastOp>();
      if (!outputShapeCastOp) {
        continue;
      }
      LDBG("outputShapeCastOp:\n" << outputShapeCastOp);
      auto contractOp =
          outputShapeCastOp.getSource().getDefiningOp<vector::ContractionOp>();
      if (!contractOp) {
        continue;
      }
      LDBG("contractOp:\n" << contractOp);
      Value acc = contractOp.getAcc();
      auto inputShapeCastOp = acc.getDefiningOp<vector::ShapeCastOp>();
      if (!inputShapeCastOp) {
        continue;
      }
      LDBG("inputShapeCastOp:\n" << inputShapeCastOp);
      Value input = inputShapeCastOp.getSource();
      auto blockArg = dyn_cast<BlockArgument>(input);
      if (!blockArg) {
        continue;
      }
      LDBG("blockArg:\n" << blockArg);
      hoistableShapeCast = std::make_pair(inputShapeCastOp, outputShapeCastOp);
      initArgIdx = blockArg.getArgNumber() - 1;
    }

    if (!hoistableShapeCast) {
      return failure();
    }
    vector::ShapeCastOp inSC = hoistableShapeCast->first;
    vector::ShapeCastOp outSC = hoistableShapeCast->second;
    SmallVector<Value> forOpInitArgs = forOp.getInitArgs();
    Value source = forOpInitArgs[initArgIdx];
    Value sourceSC =
        rewriter
            .create<vector::ShapeCastOp>(forOp.getLoc(), inSC.getType(), source)
            .getResult();
    forOpInitArgs[initArgIdx] = sourceSC;
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), forOpInitArgs);
    LDBG("newForOp:\n" << newForOp);
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(),
                         newForOp.getBody()->getArguments());
    auto newYieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    LDBG("newYieldOp:\n" << newYieldOp);
    SmallVector<Value> newForOpResults =
        newYieldOp.getOperation()->getOperands();
    int contractResultIndex;
    for (auto result : llvm::enumerate(newForOpResults)) {
      if (result.value() == outSC.getResult()) {
        newForOpResults[result.index()] = outSC.getSource();
        contractResultIndex = result.index();
      }
    }
    rewriter.updateRootInPlace(newYieldOp, [&]() {
      newYieldOp.getOperation()->setOperands(newForOpResults);
    });
    LDBG("newForOp with body:\n" << newForOp);
    SmallVector<Value> newResults = newForOp.getResults();
    Value hoistedOutputShapeCast =
        rewriter
            .create<vector::ShapeCastOp>(forOp.getLoc(), outSC.getType(),
                                         newResults[contractResultIndex])
            .getResult();
    LDBG("hoistedOutputShapeCast:\n" << hoistedOutputShapeCast);
    newResults[contractResultIndex] = hoistedOutputShapeCast;
    rewriter.replaceOp(forOp, newResults);

    return success();
  }
};

namespace {
struct LLVMCPUFoldVectorContractUnitDimsPass
    : public LLVMCPUFoldVectorContractUnitDimsBase<
          LLVMCPUFoldVectorContractUnitDimsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    vector::VectorDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LLVMCPUFoldVectorContractUnitDimsPass::runOnOperation() {
  Operation *funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet foldUnitDimsPatterns(context);
  foldUnitDimsPatterns
      .add<DropVectorContractUnitDims, HoistShapeCastOutOfSCFFor>(context);
  if (failed(applyPatternsAndFoldGreedily(funcOp,
                                          std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUFoldVectorContractUnitDimsPass() {
  return std::make_unique<LLVMCPUFoldVectorContractUnitDimsPass>();
}

void populateFoldVectorContractUnitDimsPass(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<DropVectorContractUnitDims>(context);
}

} // namespace iree_compiler
} // namespace mlir
