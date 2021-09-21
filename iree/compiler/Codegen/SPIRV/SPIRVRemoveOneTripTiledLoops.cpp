// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-remove-one-trip-tiled-loops"

namespace mlir {
namespace iree_compiler {

static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

namespace {
/// Replaces hal.interface.workgroup.count op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupCountOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupCountOp> {
 public:
  ConcretizeWorkgroupCountOp(MLIRContext *context,
                             ArrayRef<int64_t> numWorkgroups,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        numWorkgroups(numWorkgroups.begin(), numWorkgroups.end()) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex >= numWorkgroups.size()) return failure();
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, rewriter.getIndexAttr(numWorkgroups[dimIndex]));

    return success();
  }

 private:
  SmallVector<int64_t, 4> numWorkgroups;
};

// Canonicalizes away a trip-one scf.for loop by inlining its body and removing
// the loop.
//
// This pattern is needed because in Flow abstract tiling and distribution we
// will create scf.for loops that distribute workload cyclically. After
// concretizing hal.interface.workgroup.* ops, these scf.for loops still remain,
// and they will be of the form:
//
//   %lb = mul %workgroup_id_{x|y|z}, %cst_tile_size_{x|y|z}
//   scf.for %iv = %lb to %cst_wokload_size_{x|y|z}
//                 step %cst_workload_size_{x|y|z} { ... }
//
// Such scf.for loops can be inlined if %lb is smaller than upper bound.
class RemoveTripOneLoop final : public OpRewritePattern<scf::ForOp> {
 public:
  RemoveTripOneLoop(MLIRContext *context, ArrayRef<int64_t> workloadSize,
                    ArrayRef<int64_t> workloadPerWorkgroup,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize.begin(), workloadSize.end()),
        workloadPerWorkgroup(workloadPerWorkgroup.begin(),
                             workloadPerWorkgroup.end()) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // Get constant upper bound and step values.
    IntegerAttr ub, step;
    if (!matchPattern(op.upperBound(), m_Constant(&ub)) ||
        !matchPattern(op.step(), m_Constant(&step))) {
      return failure();
    }

    // Require that they are the same.
    if (ub != step) return failure();

    // Now make sure the lower bound is smaller than upper bound. The lower
    // bound should be multiplying the workgroup ID with some constant.
    auto mulOp = op.lowerBound().getDefiningOp<AffineApplyOp>();
    if (!mulOp || mulOp.mapOperands().size() != 1) return failure();

    auto operand = mulOp.mapOperands().front();
    auto idOp = operand.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
    if (!idOp) return failure();

    // We just need to make sure the max value of the workgroup ID multipled by
    // the multipler is smaller than the upper bound to guarantee one trip.
    unsigned dimIndex = idOp.dimension().getZExtValue();
    int64_t dimSize = workloadSize[dimIndex];
    if (dimSize == ShapedType::kDynamicSize) return failure();

    int64_t dimTile = workloadPerWorkgroup[dimIndex];
    AffineExpr symbol;
    bindSymbols(op.getContext(), symbol);
    auto mulMap = AffineMap::get(0, 1, symbol * dimTile);
    if (mulOp.getAffineMap() != mulMap) return failure();

    int64_t count = ceilDiv(dimSize, dimTile);
    assert(count > 0 && "expected at least one tile!");

    // ID should be in range [0, count).
    if ((count - 1) * dimTile >= ub.getInt()) {
      // Dead loop. It can actually be removed entirely. But we aren't expecting
      // it to happen here. Do not canonicalize for such case.
      return failure();
    }

    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());

    Block *block = &op.getLoopBody().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);

    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> workloadPerWorkgroup;
};
}  // namespace

static void removeOneTripTiledLoops(MLIRContext *context, FuncOp funcOp,
                                    linalg::LinalgOp rootLinalgOp,
                                    ArrayRef<int64_t> workloadPerWorkgroup) {
  unsigned numParallelDims = rootLinalgOp.getNumParallelLoops();
  unsigned numTiledDims =
      std::min<size_t>(numParallelDims, kNumMaxParallelDims);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(rootLinalgOp, 0);
  if (outputShape.size() < numParallelDims) return;

  // TODO(ravishankarm, antiagainst): Its pure co-incidence that the
  // workload is derivable from the output shape. There is no requirement
  // for this but is the case for all operations we are interested in.
  auto workloadSize = llvm::to_vector<4>(llvm::reverse(
      outputShape.take_front(numParallelDims).take_back(numTiledDims)));
  if (llvm::any_of(workloadSize, [](int64_t dim) {
        return dim == ShapedType::kDynamicSize;
      })) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Queried workload size: ";
    llvm::interleaveComma(workloadSize, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  SmallVector<int64_t, 3> numWorkgroups;
  assert(workloadPerWorkgroup.size() == workloadSize.size());
  for (auto pair : llvm::zip(workloadSize, workloadPerWorkgroup)) {
    auto workload = std::get<0>(pair);
    auto size = std::get<1>(pair);
    numWorkgroups.push_back(ceilDiv(workload, size));
  }
  numWorkgroups.resize(kNumMaxParallelDims, 1);
  {
    OwningRewritePatternList workgroupCountPatterns(context);
    workgroupCountPatterns.insert<ConcretizeWorkgroupCountOp>(context,
                                                              numWorkgroups);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(workgroupCountPatterns));
  }
  {
    OwningRewritePatternList removeTripOneLoopPatterns(context);
    removeTripOneLoopPatterns.insert<RemoveTripOneLoop>(context, workloadSize,
                                                        workloadPerWorkgroup);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(removeTripOneLoopPatterns));
  }
}

namespace {
class SPIRVRemoveOneTripTiledLoopPass
    : public SPIRVRemoveOneTripTiledLoopBase<SPIRVRemoveOneTripTiledLoopPass> {
 public:
  SPIRVRemoveOneTripTiledLoopPass() = default;
  SPIRVRemoveOneTripTiledLoopPass(const SPIRVRemoveOneTripTiledLoopPass &) {}

  void runOnOperation() {
    FuncOp funcOp = getOperation();
    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;

    // This pass seems to be only needed for the convolution vectorization. So
    // filter out the necessary conv ops.
    SmallVector<Operation *> rootOp;
    SmallVector<TiledLoopInfo> tiledLoops;
    if (failed(getFilteredOps(
            funcOp,
            [](Operation *op) {
              return isa<linalg::DepthwiseConv2DNhwOp,
                         linalg::Conv2DNhwcHwcfOp>(op);
            },
            rootOp, tiledLoops))) {
      return;
    }

    if (!llvm::hasSingleElement(rootOp)) return;
    IREE::HAL::TranslationInfo translationInfo =
        getTranslationInfo(entryPointOp);
    if (!translationInfo) return;
    ArrayAttr workloadPerWorkgroupAttr = translationInfo.workloadPerWorkgroup();
    if (!workloadPerWorkgroupAttr) return;
    auto workloadPerWorkgroup = llvm::to_vector<4>(llvm::map_range(
        workloadPerWorkgroupAttr,
        [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));

    MLIRContext *context = &getContext();
    removeOneTripTiledLoops(context, funcOp, cast<linalg::LinalgOp>(rootOp[0]),
                            workloadPerWorkgroup);
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVRemoveOneTripTiledLoopPass() {
  return std::make_unique<SPIRVRemoveOneTripTiledLoopPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
