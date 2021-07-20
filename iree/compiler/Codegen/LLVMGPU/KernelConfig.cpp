// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static constexpr unsigned cudaWarpSize = 32;

static void setConfig(TileSizesListType tileSizes, Operation* op) {
  IREE::HAL::LoweringConfig config =
      buildConfigAttr(tileSizes, ArrayRef<int64_t>{}, op->getContext());
  setLoweringConfig(op, config);
}

/// Sets the translation info on the `hal.executable.entry_point` op
/// corresponding to the `entryPointFn`. Returns failure if a translation info
/// is already set on the entry point op and is incompatible with what is being
/// set.
static LogicalResult setTranslationInfo(
    FuncOp entryPointFn, IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workloadPerWorkgroup) {
  auto entryPointOp = getEntryPoint(entryPointFn);
  auto translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointFn.getContext());
  return setTranslationInfo(entryPointOp, translationInfo);
}

static LogicalResult setRootConfig(FuncOp entryPoint, linalg::GenericOp op) {
  IREE::HAL::DispatchLoweringPassPipeline passPipeline =
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  TileSizesListType tileSizes;
  size_t numLoops = getNumOuterParallelLoops(op);
  if (numLoops == 0) {
    // Pure reduction, we serialize the operation on a single thread.
    // TODO: Use atomic to allow distributing reduction loops.
    tileSizes.push_back({});
    setConfig(tileSizes, op);
    return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
  }

  workgroupSize = {cudaWarpSize, 1, 1};
  // Pick a fixed tile size independent of the original shape.
  // TODO(thomasraoux): Currently the original shape information is lost during
  // tiling at the flow level. We need way to access it to be able to make a
  // better choice of tile size.
  int64_t lowerTs = 4 * cudaWarpSize;
  SmallVector<int64_t, 4> ts;
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  tileSizes.push_back(ts);  // Workgroup level.
  tileSizes.push_back({});  // Subgroup level.
  ts.back() = lowerTs / cudaWarpSize;
  tileSizes.push_back(ts);  // Thread level.
  setConfig(tileSizes, op);
  return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
}

static LogicalResult setRootConfig(FuncOp entryPoint, linalg::MatmulOp op) {
  IREE::HAL::DispatchLoweringPassPipeline passPipeline =
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  TileSizesListType tileSizes;
  const int64_t numWarp = 2;
  SmallVector<int64_t, 3> workgroupSize = {numWarp * cudaWarpSize, 1, 1};
  // Currently just a basic tile size to enable tiling and vectorization.
  // TODO: pick a more efficient tile size and tile at subgroup level.
  SmallVector<int64_t, 4> ts = {2, 256, 4};
  tileSizes.push_back(ts);  // Workgroup level.
  tileSizes.push_back({});  // Subgroup level.
  SmallVector<int64_t, 4> invocationLevelTs = {ts[0] / workgroupSize[1],
                                               ts[1] / workgroupSize[0]};
  tileSizes.push_back(invocationLevelTs);  // Thread level.
  setConfig(tileSizes, op);
  return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
}

static LogicalResult setRootConfig(FuncOp entryPoint,
                                   linalg::BatchMatmulOp op) {
  IREE::HAL::DispatchLoweringPassPipeline passPipeline =
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  TileSizesListType tileSizes;
  const int64_t numWarp = 2;
  SmallVector<int64_t, 3> workgroupSize = {numWarp * cudaWarpSize, 1, 1};
  SmallVector<int64_t, 4> ts = {1, 2, 256, 4};
  tileSizes.push_back(ts);  // Workgroup level.
  tileSizes.push_back({});  // Subgroup level.
  SmallVector<int64_t, 4> invocationLevelTs = {ts[0], ts[1] / workgroupSize[1],
                                               ts[2] / workgroupSize[0]};
  tileSizes.push_back(invocationLevelTs);  // Thread level.
  setConfig(tileSizes, op);
  return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(FuncOp entryPoint,
                                          linalg::LinalgOp op) {
  IREE::HAL::DispatchLoweringPassPipeline passPipeline =
      IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  TileSizesListType tileSizes;
  size_t numLoops = getNumOuterParallelLoops(op);
  if (numLoops == 0) {
    return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
  }

  workgroupSize = {cudaWarpSize, 1, 1};
  int64_t lowerTs = 4 * cudaWarpSize;
  SmallVector<int64_t, 4> ts;
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  tileSizes.push_back(ts);
  tileSizes.push_back({});  // Subgroup level.
  ts.back() = lowerTs / cudaWarpSize;
  tileSizes.push_back(ts);  // Thread level.
  setConfig(tileSizes, op);
  return setTranslationInfo(entryPoint, passPipeline, workgroupSize);
}

static LogicalResult setRootConfig(FuncOp entryPointFn,
                                   linalg::LinalgOp linalgOp) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation()))
    return setRootConfig(entryPointFn, genericOp);
  if (auto matmul = dyn_cast<linalg::MatmulOp>(linalgOp.getOperation()))
    return setRootConfig(entryPointFn, matmul);
  if (auto batchMatmul =
          dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation()))
    return setRootConfig(entryPointFn, batchMatmul);
  return setRootDefaultConfig(entryPointFn, linalgOp);
}

namespace mlir {
namespace iree_compiler {

LogicalResult initGPULaunchConfig(ModuleOp moduleOp) {
  linalg::LinalgOp rootOperation;
  auto funcOps = moduleOp.getOps<FuncOp>();
  assert(llvm::hasSingleElement(funcOps));
  FuncOp funcOp = *funcOps.begin();
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  funcOp.walk([&](linalg::LinalgOp op) { linalgOps.push_back(op); });
  if (linalgOps.empty()) {
    return ::setTranslationInfo(
        funcOp, IREE::HAL::DispatchLoweringPassPipeline::LLVMGPUDistribute,
        {1, 1, 1});
  }
  if (linalgOps.size() == 1) rootOperation = *linalgOps.begin();
  // if there is more than one linalg op, look for the root one.
  for (linalg::LinalgOp op : linalgOps) {
    if (isa<linalg::BatchMatmulOp, linalg::MatmulOp,
            linalg::ConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCOp,
            linalg::ConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCOp, linalg::PoolingNhwcMaxOp,
            linalg::PoolingNhwcMinOp, linalg::PoolingNhwcSumOp>(
            op.getOperation())) {
      rootOperation = op;
      break;
    }
  }
  if (!rootOperation) {
    // If no named ops the dispatch region should have at exactly one generic op
    // which is root operation.
    assert(llvm::count_if(linalgOps, [](linalg::LinalgOp op) {
             return isa<linalg::GenericOp>(op);
           }) == 1);
    for (linalg::LinalgOp op : linalgOps) {
      if (isa<linalg::GenericOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }
  if (failed(setRootConfig(funcOp, rootOperation))) return failure();
  IREE::HAL::LoweringConfig config = getLoweringConfig(rootOperation);
  for (linalg::LinalgOp op : linalgOps) {
    if (op == rootOperation) continue;
    setLoweringConfig(op, config);
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
