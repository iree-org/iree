// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_LOWERARGCOMPARETOVECTORPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

static Value cloneComparatorRegion(RewriterBase &rewriter, Region &region,
                                   Value lhs, Value rhs) {
  Block &block = region.front();
  IRMapping mapper;
  mapper.map(block.getArgument(0), lhs);
  mapper.map(block.getArgument(1), rhs);
  for (Operation &op : block.without_terminator()) {
    Operation *clonedOp = rewriter.clone(op, mapper);
    for (const auto &[origResult, clonedResult] :
         llvm::zip_equal(op.getResults(), clonedOp->getResults())) {
      mapper.map(origResult, clonedResult);
    }
  }
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  return mapper.lookup(yieldOp.getValues()[0]);
}

static SmallVector<int64_t> delinearizeIndex(int64_t linearIdx,
                                             ArrayRef<int64_t> shape) {
  SmallVector<int64_t> indices;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    indices.push_back(linearIdx % shape[i]);
    linearIdx /= shape[i];
  }
  std::reverse(indices.begin(), indices.end());
  return indices;
}

static SmallVector<int64_t> buildMoveToLastPerm(int64_t rank, int64_t dim) {
  SmallVector<int64_t> perm;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      perm.push_back(i);
    }
  }
  perm.push_back(dim);
  return perm;
}

struct LowerArgCompareToVector final : OpRewritePattern<ArgCompareOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(ArgCompareOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType inputType = op.getInputValueType();

    if (inputType.isScalable()) {
      return rewriter.notifyMatchFailure(op, "scalable vectors not supported");
    }

    int64_t rank = inputType.getRank();
    int64_t reductionDim = op.getDimension();
    int64_t reductionSize = inputType.getShape()[reductionDim];

    Value inputValue = op.getInputValue();
    Value inputIndex = op.getInputIndex();
    Value initValue = op.getInitValue();
    Value initIndex = op.getInitIndex();
    Value indexBase = op.getIndexBase();
    Type indexElemTy = op.getInitIndexElementType();

    bool needsTranspose = (reductionDim != rank - 1) && (rank > 1);
    if (needsTranspose) {
      SmallVector<int64_t> perm = buildMoveToLastPerm(rank, reductionDim);
      inputValue = vector::TransposeOp::create(rewriter, loc, inputValue, perm);
      if (inputIndex) {
        inputIndex =
            vector::TransposeOp::create(rewriter, loc, inputIndex, perm);
      }
      reductionDim = rank - 1;
    }

    VectorType transposedType = cast<VectorType>(inputValue.getType());
    ArrayRef<int64_t> shape = transposedType.getShape();

    SmallVector<int64_t> parallelShape(shape.begin(), shape.end() - 1);
    int64_t numParallelElems = ShapedType::getNumElements(parallelShape);

    VectorType resultValueType = op.getInitValueType();
    VectorType resultIndexType = op.getInitIndexType();

    Value resultValue = initValue;
    Value resultIndex = initIndex;

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cRedSize =
        arith::ConstantIndexOp::create(rewriter, loc, reductionSize);

    for (int64_t p = 0; p < numParallelElems; ++p) {
      SmallVector<int64_t> parallelIndices =
          delinearizeIndex(p, parallelShape);

      Value accVal, accIdx;
      if (resultValueType.getRank() == 0) {
        accVal = vector::ExtractOp::create(rewriter, loc, initValue,
                                           ArrayRef<int64_t>{});
        accIdx = vector::ExtractOp::create(rewriter, loc, initIndex,
                                           ArrayRef<int64_t>{});
      } else {
        accVal = vector::ExtractOp::create(rewriter, loc, initValue,
                                           parallelIndices);
        accIdx = vector::ExtractOp::create(rewriter, loc, initIndex,
                                           parallelIndices);
      }

      Value inputSlice, indexSlice;
      if (rank == 1) {
        inputSlice = inputValue;
        if (inputIndex) {
          indexSlice = inputIndex;
        }
      } else {
        inputSlice = vector::ExtractOp::create(rewriter, loc, inputValue,
                                               parallelIndices);
        if (inputIndex) {
          indexSlice = vector::ExtractOp::create(rewriter, loc, inputIndex,
                                                 parallelIndices);
        }
      }

      auto forOp = scf::ForOp::create(rewriter, loc, c0, cRedSize, c1,
                                      ValueRange{accVal, accIdx});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value bestVal = forOp.getRegionIterArg(0);
        Value bestIdx = forOp.getRegionIterArg(1);

        Value candidate =
            vector::ExtractOp::create(rewriter, loc, inputSlice, iv);

        Value cmpResult =
            cloneComparatorRegion(rewriter, op.getRegion(), candidate, bestVal);

        Value newVal = arith::SelectOp::create(rewriter, loc, cmpResult,
                                               candidate, bestVal);

        Value candidateIdx;
        if (inputIndex) {
          candidateIdx =
              vector::ExtractOp::create(rewriter, loc, indexSlice, iv);
        } else {
          candidateIdx = iv;
          if (indexBase) {
            candidateIdx =
                arith::AddIOp::create(rewriter, loc, indexBase, candidateIdx);
          }
          if (candidateIdx.getType() != indexElemTy) {
            candidateIdx = arith::IndexCastOp::create(
                rewriter, loc, indexElemTy, candidateIdx);
          }
        }

        Value newIdx = arith::SelectOp::create(rewriter, loc, cmpResult,
                                               candidateIdx, bestIdx);

        scf::YieldOp::create(rewriter, loc, ValueRange{newVal, newIdx});
      }

      Value reducedVal = forOp.getResult(0);
      Value reducedIdx = forOp.getResult(1);

      if (resultValueType.getRank() == 0) {
        resultValue = vector::BroadcastOp::create(rewriter, loc,
                                                  resultValueType, reducedVal);
        resultIndex = vector::BroadcastOp::create(rewriter, loc,
                                                  resultIndexType, reducedIdx);
      } else {
        resultValue = vector::InsertOp::create(rewriter, loc, reducedVal,
                                               resultValue, parallelIndices);
        resultIndex = vector::InsertOp::create(rewriter, loc, reducedIdx,
                                               resultIndex, parallelIndices);
      }
    }

    rewriter.replaceOp(op, ValueRange{resultValue, resultIndex});
    return success();
  }
};

struct LowerArgCompareToVectorPass final
    : impl::LowerArgCompareToVectorPassBase<LowerArgCompareToVectorPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerArgCompareToVector>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateLowerArgCompareToVectorPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerArgCompareToVector>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
