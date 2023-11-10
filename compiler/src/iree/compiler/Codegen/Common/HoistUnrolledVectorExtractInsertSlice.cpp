// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-hoist-unrolled-vector"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {

/// Returns all the users of `srcTensor` if they are artifacts from vector
/// unrolling. It is true only if
///   1. All the users are vector.extract_strided_slice ops.
///   2. Each vector.insert_strided_slice can map to a
///      vector.extract_stirded_slice op.
/// Returns failure if it can not find the set from vector unrolling artifacts.
static FailureOr<SmallVector<vector::ExtractStridedSliceOp>>
getUnrolledExtractSlices(BlockArgument srcTensor,
                         SmallVector<vector::InsertStridedSliceOp> insertOps) {
  SmallVector<vector::ExtractStridedSliceOp> res;
  for (auto user : srcTensor.getUsers()) {
    auto extractStridedSliceOp = dyn_cast<vector::ExtractStridedSliceOp>(user);
    if (!extractStridedSliceOp)
      return failure();
    res.push_back(extractStridedSliceOp);
  }
  if (res.size() != insertOps.size())
    return failure();

  std::reverse(res.begin(), res.end());
  for (const auto [extractOp, insertOp] : llvm::zip_equal(res, insertOps)) {
    auto offset0 = insertOp.getOffsets();
    auto offset1 = extractOp.getOffsets();
    if (offset0 != offset1)
      return failure();
  }

  return res;
}

/// Returns the vector.insert_strided_slice ops from vector unrolling. They are
/// artifact only if
///   1. A vector.insert_strided_slice op is yielded by scf.yield op.
///   2. The sequence of vector.insert_strided_slice op exactly covers the
///      yielded vector.
/// Returns failure if the set can not be found.
static FailureOr<SmallVector<vector::InsertStridedSliceOp>>
getUnrolledInsertSlices(scf::ForOp forOp, BlockArgument bbArg,
                        OpOperand &yieldOperand) {
  assert(bbArg.getArgNumber() ==
             forOp.getNumInductionVars() + yieldOperand.getOperandNumber() &&
         "bbArg and yieldOperand must match");
  assert(isa<scf::YieldOp>(yieldOperand.getOwner()) && "must be an scf.yield");

  SmallVector<vector::InsertStridedSliceOp> res;
  Value v = yieldOperand.get();
  auto insertStridedSliceOp = v.getDefiningOp<vector::InsertStridedSliceOp>();
  if (!insertStridedSliceOp)
    return failure();

  ArrayRef<int64_t> vecShape =
      insertStridedSliceOp.getSourceVectorType().getShape();
  ArrayRef<int64_t> destShape =
      insertStridedSliceOp.getDestVectorType().getShape();
  int numOps = 1;
  for (auto [vecSize, destSize] : llvm::zip_equal(vecShape, destShape)) {
    if (destSize % vecSize)
      return failure();
    numOps *= destSize / vecSize;
  }

  while (insertStridedSliceOp) {
    res.push_back(insertStridedSliceOp);
    insertStridedSliceOp = insertStridedSliceOp.getDest()
                               .getDefiningOp<vector::InsertStridedSliceOp>();
  }
  if (res.size() != numOps)
    return failure();

  std::reverse(res.begin(), res.end());
  SmallVector<int64_t> expectedOffsets(vecShape.size(), 0);
  for (vector::InsertStridedSliceOp op : res) {
    SmallVector<int64_t> offsets = getI64SubArray(op.getOffsets());
    if (expectedOffsets != offsets)
      return failure();
    expectedOffsets.back() += vecShape.back();
    for (int pos = expectedOffsets.size() - 1; pos > 0; pos--) {
      if (expectedOffsets[pos] != destShape[pos])
        break;
      expectedOffsets[pos] = 0;
      expectedOffsets[pos - 1] += vecShape[pos - 1];
    }
  }
  return res;
}

static scf::ForOp hoistVectorExtractInsertSlice(
    RewriterBase &rewriter,
    SmallVectorImpl<vector::ExtractStridedSliceOp> &extractOps,
    SmallVectorImpl<vector::InsertStridedSliceOp> &insertOps,
    BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  int64_t initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // 1. Hoist all the read ops. This will not trigger dominance violations once
  // BBArgs are updated.
  for (auto extractStridedSliceOp : extractOps) {
    extractStridedSliceOp->moveBefore(forOp);
    if (!forOp.isDefinedOutsideOfLoop(extractStridedSliceOp.getVector())) {
      assert(extractStridedSliceOp.getVector() == tensorBBArg &&
             "extractSlice source not defined above must be the tracked bbArg");
      rewriter.startRootUpdate(extractStridedSliceOp);
      extractStridedSliceOp.getVectorMutable().assign(
          forOp.getInitArgs()[initArgNumber]);
      rewriter.finalizeRootUpdate(extractStridedSliceOp);
    }
  }

  // 2. Rewrite `loop` with an additional yield. This is the quantity that is
  // computed iteratively but whose storage has become loop-invariant.
  NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                 ArrayRef<BlockArgument> newBBArgs) {
    return llvm::map_to_vector(insertOps,
                               [](auto v) -> Value { return v.getSource(); });
  };
  SmallVector<Value> extractResults = llvm::map_to_vector(
      extractOps, [](auto v) -> Value { return v.getResult(); });
  auto newForOp = cast<scf::ForOp>(*forOp.replaceWithAdditionalYields(
      rewriter, extractResults, /*replaceInitOperandUsesInLoop=*/true,
      yieldFn));

  // 3. Update the yield. Invariant: initArgNumber is the destination tensor.
  auto yieldOp =
      cast<scf::YieldOp>(newForOp.getRegion().front().getTerminator());
  rewriter.startRootUpdate(yieldOp);
  yieldOp->setOperand(initArgNumber, insertOps[0].getDest());
  rewriter.finalizeRootUpdate(yieldOp);

  // 4. Hoist all the write ops after and make uses of
  // newForOp.getResult(initArgNumber) flow through it.
  for (auto [idx, insertStridedSliceOp] : llvm::enumerate(insertOps)) {
    insertStridedSliceOp->moveAfter(newForOp);
    rewriter.startRootUpdate(insertStridedSliceOp);
    insertStridedSliceOp.getSourceMutable().assign(
        newForOp.getResults()[initArgNumber + idx + 1]);
    insertStridedSliceOp.getDestMutable().assign(
        newForOp.getResults()[initArgNumber]);
    rewriter.finalizeRootUpdate(insertStridedSliceOp);
    rewriter.replaceAllUsesExcept(newForOp.getResult(initArgNumber),
                                  insertStridedSliceOp.getResult(),
                                  insertStridedSliceOp);
  }
  return newForOp;
}

static scf::ForOp hoistUnrolledVectorExtractInsert(RewriterBase &rewriter,
                                                   scf::ForOp forOp) {
  LLVM_DEBUG(DBGS() << "Enter hoistRedundantSubsetExtractInsert scf.for\n");
  Operation *yield = forOp.getBody()->getTerminator();

  LLVM_DEBUG(DBGS() << "\n"; DBGS() << "Consider " << forOp << "\n");

  scf::ForOp newForOp = forOp;
  do {
    forOp = newForOp;
    for (const auto &it : llvm::enumerate(forOp.getRegionIterArgs())) {
      LLVM_DEBUG(DBGS() << "Consider " << it.value() << "\n");
      OpOperand &ret = yield->getOpOperand(it.index());
      auto insertOps = getUnrolledInsertSlices(forOp, it.value(), ret);
      if (failed(insertOps))
        continue;
      auto extractOps = getUnrolledExtractSlices(it.value(), insertOps.value());
      if (failed(extractOps))
        continue;
      newForOp = hoistVectorExtractInsertSlice(rewriter, extractOps.value(),
                                               insertOps.value(), it.value());
      break;
    }
  } while (forOp != newForOp);

  return newForOp;
}

namespace {
class HoistUnrolledVectorExtractInsertSlicePass
    : public HoistUnrolledVectorExtractInsertSliceBase<
          HoistUnrolledVectorExtractInsertSlicePass> {
public:
  using HoistUnrolledVectorExtractInsertSliceBase::
      HoistUnrolledVectorExtractInsertSliceBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void HoistUnrolledVectorExtractInsertSlicePass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());
  funcOp.walk([&](scf::ForOp forOp) {
    hoistUnrolledVectorExtractInsert(rewriter, forOp);
  });

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
  vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistUnrolledVectorExtractInsertSlicePass() {
  return std::make_unique<HoistUnrolledVectorExtractInsertSlicePass>();
}

} // namespace iree_compiler
} // namespace mlir
