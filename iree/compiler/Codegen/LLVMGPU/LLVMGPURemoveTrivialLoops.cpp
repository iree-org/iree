// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
static Optional<std::pair<AffineExpr, AffineExpr>> getWorkgroupRange(
    Value processorValue, SmallVectorImpl<Value> & /*dims*/,
    SmallVectorImpl<Value> & /*symbols*/, ArrayRef<int64_t> workgroupCount,
    ArrayRef<int64_t> workgroupSize) {
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = StringSwitch<unsigned>(idOp.dimension())
                         .Case("x", 0)
                         .Case("y", 1)
                         .Case("z", 2);
    OpBuilder b(processorValue.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
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

/// Infer the number of workgroups by looking at the tiled loop and the number
/// of element per workgroups.
static SmallVector<int64_t> getNumWorkgroup(
    FuncOp funcOp, IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  SmallVector<TiledLoopInfo> tiledLoopInfo = getTiledLoopInfo(funcOp);
  SmallVector<int64_t> workloadSize(tiledLoopInfo.size());
  for (TiledLoopInfo &tileInfo : tiledLoopInfo) {
    if (tileInfo.distributionDim >= workloadSize.size())
      return SmallVector<int64_t>();
    if (!tileInfo.lb.is<Attribute>() || !tileInfo.ub.is<Attribute>() ||
        !tileInfo.step.is<Attribute>()) {
      continue;
    }
    int64_t lb = tileInfo.lb.get<Attribute>().cast<IntegerAttr>().getInt();
    int64_t ub = tileInfo.ub.get<Attribute>().cast<IntegerAttr>().getInt();
    int64_t step = tileInfo.step.get<Attribute>().cast<IntegerAttr>().getInt();
    if (step == 0) return SmallVector<int64_t>();
    workloadSize[tileInfo.distributionDim] = (ub - lb) / step;
  }
  auto translationInfo = getTranslationInfo(entryPointOp);
  if (!translationInfo) return SmallVector<int64_t>();

  ArrayAttr workloadPerWorkgroupAttr = translationInfo.workloadPerWorkgroup();
  if (!workloadPerWorkgroupAttr) return SmallVector<int64_t>();
  auto workloadPerWorkgroup = llvm::to_vector<4>(llvm::map_range(
      workloadPerWorkgroupAttr,
      [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
  if (workloadSize.size() != workloadPerWorkgroup.size())
    return SmallVector<int64_t>();
  SmallVector<int64_t> numWorkgroups;
  for (auto pair : llvm::zip(workloadSize, workloadPerWorkgroup)) {
    auto workload = std::get<0>(pair);
    auto size = std::get<1>(pair);
    numWorkgroups.push_back(llvm::divideCeil(workload, size));
  }
  numWorkgroups.resize(kNumMaxParallelDims, 1);
  return numWorkgroups;
}

static void removeOneTripTiledLoops(FuncOp funcOp,
                                    ArrayRef<int64_t> workgroupSize,
                                    ArrayRef<int64_t> numWorkgroups) {
  auto getWorkgroupRangeFn = [numWorkgroups, workgroupSize](
                                 Value processorValue,
                                 SmallVectorImpl<Value> &dims,
                                 SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups,
                             workgroupSize);
  };
  OwningRewritePatternList patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class LLVMGPURemoveSingleIterationLoopPass
    : public LLVMGPURemoveSingleIterationLoopBase<
          LLVMGPURemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;

    SmallVector<int64_t> workgroupSize = getWorkgroupSize(entryPointOp);
    SmallVector<int64_t> numWorkgroups = getNumWorkgroup(funcOp, entryPointOp);

    removeOneTripTiledLoops(funcOp, workgroupSize, numWorkgroups);
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createLLVMGPURemoveSingleIterationLoopPass() {
  return std::make_unique<LLVMGPURemoveSingleIterationLoopPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
