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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-remove-trivial-loops"

namespace mlir {
namespace iree_compiler {

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
    case gpu::Dimension::x:
      return 0;
    case gpu::Dimension::y:
      return 1;
    case gpu::Dimension::z:
      return 2;
    default:
      assert(false && "invalid dimension");
      return 0;
  }
}

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
static Optional<std::pair<AffineExpr, AffineExpr>> getWorkgroupRange(
    Value processorValue, SmallVectorImpl<Value> & /*dims*/,
    SmallVectorImpl<Value> & /*symbols*/, ArrayRef<int64_t> workgroupCount,
    ArrayRef<int64_t> workgroupSize) {
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = dimToIndex(idOp.dimension());
    OpBuilder b(processorValue.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = dimToIndex(dimOp.dimension());
    AffineExpr bound = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(bound, bound);
  }

  if (workgroupCount.empty()) return llvm::None;

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

/// Return true if the given tiled loop is distributed to workgroups.
static bool isWorkgroupLoop(const LoopTilingAndDistributionInfo &info) {
  auto forOp = cast<scf::ForOp>(info.loop);
  Operation *lbOp = forOp.getLowerBound().getDefiningOp();
  if (isa<IREE::HAL::InterfaceWorkgroupIDOp>(lbOp)) return true;
  auto applyOp = dyn_cast<AffineApplyOp>(lbOp);
  return applyOp && llvm::any_of(applyOp.getMapOperands(), [](Value operand) {
           return operand.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
         });
}

/// Infer the number of workgroups by looking at the tiled loop and the number
/// of element per workgroups.
static SmallVector<int64_t> getNumWorkgroup(
    func::FuncOp funcOp, IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  auto allLoops = getTiledAndDistributedLoopInfo(funcOp);
  auto wgLoops =
      llvm::to_vector<3>(llvm::make_filter_range(allLoops, isWorkgroupLoop));
  SmallVector<int64_t> workloadSize(wgLoops.size());
  for (LoopTilingAndDistributionInfo &tileInfo : wgLoops) {
    if (tileInfo.processorDistributionDim >= workloadSize.size()) return {};
    if (!tileInfo.untiledLowerBound.is<Attribute>() ||
        !tileInfo.untiledUpperBound.is<Attribute>() ||
        !tileInfo.untiledStep.is<Attribute>()) {
      continue;
    }
    int64_t lb = tileInfo.untiledLowerBound.get<Attribute>()
                     .cast<IntegerAttr>()
                     .getInt();
    int64_t ub = tileInfo.untiledUpperBound.get<Attribute>()
                     .cast<IntegerAttr>()
                     .getInt();
    int64_t step =
        tileInfo.untiledStep.get<Attribute>().cast<IntegerAttr>().getInt();
    if (step == 0) return SmallVector<int64_t>();
    workloadSize[tileInfo.processorDistributionDim] = (ub - lb) / step;
  }
  auto translationInfo = getTranslationInfo(entryPointOp);
  if (!translationInfo) return SmallVector<int64_t>();

  SmallVector<int64_t> workloadPerWorkgroup =
      translationInfo.getWorkloadPerWorkgroupVals();
  if (workloadSize.size() != workloadPerWorkgroup.size()) {
    return SmallVector<int64_t>();
  }
  SmallVector<int64_t> numWorkgroups;
  for (auto pair : llvm::zip(workloadSize, workloadPerWorkgroup)) {
    auto workload = std::get<0>(pair);
    auto size = std::get<1>(pair);
    numWorkgroups.push_back(llvm::divideCeil(workload, size));
  }
  numWorkgroups.resize(kNumMaxParallelDims, 1);
  return numWorkgroups;
}

static LogicalResult removeOneTripTiledLoops(func::FuncOp funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             ArrayRef<int64_t> numWorkgroups) {
  auto getWorkgroupRangeFn = [numWorkgroups, workgroupSize](
                                 Value processorValue,
                                 SmallVectorImpl<Value> &dims,
                                 SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups,
                             workgroupSize);
  };
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveSingleIterationLoopPass final
    : public RemoveSingleIterationLoopBase<RemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;

    SmallVector<int64_t> workgroupSize = getWorkgroupSize(entryPointOp);
    SmallVector<int64_t> numWorkgroups = getNumWorkgroup(funcOp, entryPointOp);

    if (failed(removeOneTripTiledLoops(funcOp, workgroupSize, numWorkgroups))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass() {
  return std::make_unique<RemoveSingleIterationLoopPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
