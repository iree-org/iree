// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-remove-one-trip-tiled-loops"

namespace mlir {
namespace iree_compiler {

/// Returns the corresponding range for the given `processorValue` is a
/// workgroup ID or count op.
static Optional<std::pair<AffineExpr, AffineExpr>> getWorkgroupRange(
    Value processorValue, SmallVectorImpl<Value> & /*dims*/,
    SmallVectorImpl<Value> & /*symbols*/, ArrayRef<int64_t> workgroupCount) {
  if (auto idOp =
          processorValue.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = idOp.dimension().getZExtValue();
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue
                       .getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = dimOp.dimension().getZExtValue();
    AffineExpr bound = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(bound, bound);
  }
  return llvm::None;
}

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
  if (llvm::any_of(workloadSize, ShapedType::isDynamic)) return;

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
    numWorkgroups.push_back(llvm::divideCeil(workload, size));
  }
  numWorkgroups.resize(kNumMaxParallelDims, 1);

  auto getWorkgroupRangeFn = [numWorkgroups](Value processorValue,
                                             SmallVectorImpl<Value> &dims,
                                             SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups);
  };
  OwningRewritePatternList patterns(context);
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    auto isConvOp = [](Operation *op) {
      return isa<linalg::DepthwiseConv2DNhwOp, linalg::Conv2DNhwcHwcfOp>(op);
    };
    if (failed(getFilteredOps(funcOp, isConvOp, rootOp, tiledLoops))) return;

    if (!llvm::hasSingleElement(rootOp)) return;
    auto translationInfo = getTranslationInfo(entryPointOp);
    if (!translationInfo) return;

    auto workloadPerWorkgroup = translationInfo.getWorkloadPerWorkgroupVals();

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
