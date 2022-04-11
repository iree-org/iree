// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-insert-workgroups-count-region"

namespace mlir {
namespace iree_compiler {

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

/// Compute the workload per workgroup to use based on the tile sizes passed.
static SmallVector<int64_t> getWorkloadPerWorkgroup(
    ArrayRef<int64_t> tileSizes) {
  // TODO(ravishankarm): This for now assumes that we can just drop all the
  // zero-dim tile sizes. We need to eventually change this so that we dont have
  // to do this. It is implicity linked to the dispatch region workload having
  // the consistent information. That needs to be changed to take the entire
  // iteration domain size as the argument, and then we can use the distribute
  // loop tile sizes directly.
  SmallVector<int64_t> nonZeroTileSizes;
  for (auto ts : tileSizes) {
    if (!ts) continue;
    nonZeroTileSizes.push_back(ts);
  }
  return llvm::to_vector(llvm::reverse(nonZeroTileSizes));
}

/// Defines the workgroup count region if the tile size for the distributed
/// loops are known.
/// IREE considers it without interchange at Flow level, so the arguments are
/// not interchanged. In this context, we interchange the final yield values.
static LogicalResult defineWorkgroupCountRegion(
    func::FuncOp entryPointFn, ArrayRef<int64_t> workloadPerWorkgroup,
    ArrayRef<int64_t> interchange) {
  // TODO(hanchung): Look into partitionable loops to see if the configuraion
  // is reasonable. The loops are paritioned at Flow level, any interchange on
  // non-paritioned loops might triggers issues. It's not easy to check
  // because the information is carried on op. It's much easier when we embed
  // make partitionable loops also part of the lowering config. Otherwise, the
  // logic should be considered everywhere.
  if (interchange.size() > kNumMaxParallelDims) {
    return entryPointFn.emitOpError(
               "expected the number of loops is not greater than")
           << kNumMaxParallelDims;
  }

  if (workloadPerWorkgroup.size() > kNumMaxParallelDims) {
    // For now error out here.
    return entryPointFn.emitOpError(
               "expected workload per workgroup to be less than or equal to ")
           << kNumMaxParallelDims;
  }
  WorkgroupCountRegionBuilder regionBuilder =
      [&workloadPerWorkgroup, &interchange](
          OpBuilder &b, Location loc,
          std::array<Value, 3> workload) -> std::array<Value, 3> {
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    std::array<Value, 3> numWorkgroups = {one, one, one};
    for (auto it : llvm::enumerate(workloadPerWorkgroup)) {
      // If tile size is 0, it implies this isnt tiled, and the number of
      // workgroups is 1, i.e. the default.
      if (it.value() == 0) continue;
      numWorkgroups[it.index()] = applyMapToValues(
          b, loc,
          AffineMap::get(0, 1, b.getAffineSymbolExpr(0).ceilDiv(it.value())),
          workload[it.index()])[0];
    }
    if (interchange.empty()) return numWorkgroups;

    std::array<Value, 3> res = numWorkgroups;
    for (auto en : llvm::enumerate(interchange)) {
      res[en.value()] = numWorkgroups[en.index()];
    }
    return res;
  };
  OpBuilder builder(entryPointFn.getContext());
  return defineWorkgroupCountRegion(builder, entryPointFn, regionBuilder);
}

/// Update the workload_per_wg value on the TranslationInfoAttr.
// TODO(ravishankarm): The workload_per_wg field should be deprecated. This
// is just transition before all dependencies on it can be removed.
static LogicalResult updateTranslationInfoAttr(
    func::FuncOp entryPointFn, ArrayRef<int64_t> workloadPerWorkgroup) {
  auto entryPointOp = getEntryPoint(entryPointFn);
  if (!entryPointOp) {
    return entryPointFn.emitOpError("expected entry point function");
  }
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault;
  if (auto translationInfo = getTranslationInfo(entryPointOp)) {
    // Expect the `workload_per_wg` to be empty.
    if (!translationInfo.getWorkloadPerWorkgroupVals().empty()) {
      return entryPointFn.emitOpError(
          "expected workload_per_wg to be empty at this stage");
    }
    passPipeline = translationInfo.getDispatchLoweringPassPipeline();
  }
  auto newTranslationInfoAttr = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn.getContext(), passPipeline, workloadPerWorkgroup);
  setTranslationInfo(entryPointOp, newTranslationInfoAttr);
  return success();
}

namespace {
struct InsertDistributionInfoPass
    : public InsertDistributionInfoBase<InsertDistributionInfoPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect,
                    IREE::Flow::FlowDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void InsertDistributionInfoPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!isEntryPoint(funcOp)) return;

  SmallVector<Operation *> computeOps;
  SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
  if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
    return signalPassFailure();
  }
  if (!tiledLoops.empty()) {
    // The entry point already has distribution to workgroups. Do nothing.
    return;
  }
  if (computeOps.empty()) {
    // Ignore other operations.
    return;
  }

  // Get the tile sizes to use from lowering configuration if set.
  SmallVector<int64_t> tileSizes, interchange;
  if (failed(getDistributionTileConfigFromLoweringConfig(computeOps, tileSizes,
                                                         interchange))) {
    return signalPassFailure();
  }
  SmallVector<int64_t> workloadPerWorkroup = getWorkloadPerWorkgroup(tileSizes);

  if (failed(defineWorkgroupCountRegion(funcOp, workloadPerWorkroup,
                                        interchange)) ||
      failed(updateTranslationInfoAttr(funcOp, workloadPerWorkroup))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createInsertDistributionInfoPass() {
  return std::make_unique<InsertDistributionInfoPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
