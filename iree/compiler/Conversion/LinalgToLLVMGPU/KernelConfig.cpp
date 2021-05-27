// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/LinalgToLLVMGPU/KernelConfig.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static constexpr unsigned cudaWarpSize = 32;

static LaunchConfig getOpLaunchConfig(linalg::GenericOp op) {
  LaunchConfig config;
  size_t numLoops = getNumOuterParallelLoops(op);
  if (numLoops == 0) {
    // Pure reduction, we serialize the operation on a single thread.
    // TODO: Use atomic to allow distributing reduction loops.
    config.setWorkgroupSize({1, 1, 1});
    config.setTileSizes(op, {}, 0);
    return config;
  }

  config.setWorkgroupSize({cudaWarpSize, 1, 1});
  // Pick a fixed tile size independent of the original shape.
  // TODO(thomasraoux): Currently the original shape information is lost during
  // tiling at the flow level. We need way to access it to be able to make a
  // better choice of tile size.
  int64_t lowerTs = 4 * cudaWarpSize;
  SmallVector<int64_t, 4> ts;
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  config.setTileSizes(op, ts, 0);  // Workgroup level.
  config.setTileSizes(op, {}, 1);  // Subgroup level.
  ts.back() = lowerTs / cudaWarpSize;
  config.setTileSizes(op, ts, 2);  // Thread level.
  return config;
}

static LaunchConfig getOpLaunchConfig(linalg::MatmulOp op) {
  LaunchConfig config;
  const int64_t numWarp = 2;
  std::array<int64_t, 3> workgroupSize = {numWarp * cudaWarpSize, 1, 1};
  config.setWorkgroupSize(workgroupSize);
  // Currently just a basic tile size to enable tiling and vectorization.
  // TODO: pick a more efficient tile size and tile at subgroup level.
  SmallVector<int64_t, 4> ts = {2, 256, 4};
  config.setTileSizes(op, ts, 0);  // Workgroup level.
  config.setTileSizes(op, {}, 1);  // Subgroup level.
  SmallVector<int64_t, 4> invocationLevelTs = {ts[0] / workgroupSize[1],
                                               ts[1] / workgroupSize[0]};
  config.setTileSizes(op, invocationLevelTs, 2);  // Thread level.
  return config;
}

static LaunchConfig getOpLaunchConfig(linalg::BatchMatmulOp op) {
  LaunchConfig config;
  const int64_t numWarp = 2;
  std::array<int64_t, 3> workgroupSize = {numWarp * cudaWarpSize, 1, 1};
  config.setWorkgroupSize(workgroupSize);
  SmallVector<int64_t, 4> ts = {1, 2, 256, 4};
  config.setTileSizes(op, ts, 0);  // Workgroup level.
  config.setTileSizes(op, {}, 1);  // Subgroup level.
  SmallVector<int64_t, 4> invocationLevelTs = {ts[0], ts[1] / workgroupSize[1],
                                               ts[2] / workgroupSize[0]};
  config.setTileSizes(op, invocationLevelTs, 2);  // Thread level.
  return config;
}

// Basic default properties for linalg ops that haven't been tuned.
static LaunchConfig getDefaultOpLaunchConfig(linalg::LinalgOp op) {
  LaunchConfig config;
  size_t numLoops = getNumOuterParallelLoops(op);
  if (numLoops == 0) return config;

  config.setWorkgroupSize({cudaWarpSize, 1, 1});
  int64_t lowerTs = 4 * cudaWarpSize;
  SmallVector<int64_t, 4> ts;
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  config.setTileSizes(op, ts, 0);  // Workgroup level.
  config.setTileSizes(op, {}, 1);  // Subgroup level.
  ts.back() = lowerTs / cudaWarpSize;
  config.setTileSizes(op, ts, 2);  // Thread level.
  return config;
}

static LaunchConfig getOpLaunchConfig(linalg::LinalgOp linalgOp) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation()))
    return getOpLaunchConfig(genericOp);
  if (auto matmul = dyn_cast<linalg::MatmulOp>(linalgOp.getOperation()))
    return getOpLaunchConfig(matmul);
  if (auto batchMatmul =
          dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation()))
    return getOpLaunchConfig(batchMatmul);
  return getDefaultOpLaunchConfig(linalgOp);
}

namespace mlir {
namespace iree_compiler {

Optional<LaunchConfig> getLLVMGPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig launchConfig;

  linalg::LinalgOp rootOperation;
  if (linalgOps.empty()) return llvm::None;
  if (linalgOps.size() == 1) rootOperation = *linalgOps.begin();
  // if there is more than one linalg op, look for the root one.
  for (linalg::LinalgOp op : linalgOps) {
    if (isa<linalg::BatchMatmulOp, linalg::MatmulOp,
            linalg::ConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCOp,
            linalg::ConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCFOp,
            linalg::DepthwiseConvInputNHWCFilterHWCOp,
            linalg::PoolingNHWCMaxI8Op, linalg::PoolingNHWCMaxI16Op,
            linalg::PoolingNHWCMaxI32Op, linalg::PoolingNHWCMaxFOp,
            linalg::PoolingNHWCMinFOp, linalg::PoolingNHWCSumFOp>(
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
  launchConfig = getOpLaunchConfig(rootOperation);
  launchConfig.setRootOperation(rootOperation.getOperation());
  if (!launchConfig.getTileSizes(rootOperation, 0).empty()) {
    if (failed(propogateRootOperationLaunchConfig(launchConfig, rootOperation,
                                                  dependenceGraph)))
      return llvm::None;
  }
  return launchConfig;
}

}  // namespace iree_compiler
}  // namespace mlir
