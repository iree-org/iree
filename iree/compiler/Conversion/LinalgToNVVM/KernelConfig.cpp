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

#include "iree/compiler/Conversion/LinalgToNVVM/KernelConfig.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static constexpr unsigned cudaWarpSize = 32;

/// Fills `inputTypes` and `outputTypes` with the original input/output types
/// for all tiles for `op`.
/// Copied from iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.cpp
/// This should be moved to a common location if still needed in the future.
static void getInputOutputTypes(linalg::LinalgOp op,
                                SmallVectorImpl<ShapedType> &inputTypes,
                                SmallVectorImpl<ShapedType> &outputTypes) {
  // NOTE: Special treatment to let the flow.dispatch.workgroups path to be able
  // to query launch configurations. This should be cleaned up after the
  // flow.dispatch.workgroups become the default path.
  auto inputTypeAttr =
      op->getAttrOfType<ArrayAttr>("iree.codegen.original_input_types");
  auto outputTypeAttr =
      op->getAttrOfType<ArrayAttr>("iree.codegen.original_output_types");
  if (outputTypeAttr && inputTypeAttr) {
    for (Type type : inputTypeAttr.getAsValueRange<TypeAttr>())
      inputTypes.push_back(type.cast<ShapedType>());
    for (Type type : outputTypeAttr.getAsValueRange<TypeAttr>())
      outputTypes.push_back(type.cast<ShapedType>());
  } else {
    for (Type type : op.getInputBufferTypes())
      inputTypes.push_back(type.cast<ShapedType>());
    for (Type type : op.getOutputBufferTypes())
      outputTypes.push_back(type.cast<ShapedType>());
  }
}

static LaunchConfig getOpLaunchConfig(linalg::GenericOp op) {
  LaunchConfig config;
  size_t numLoops = getNumOuterParallelLoops(op);
  if (numLoops == 0) return config;

  SmallVector<ShapedType, 4> inputTypes, outputTypes;
  getInputOutputTypes(op, inputTypes, outputTypes);

  config.setWorkgroupSize({cudaWarpSize, 1, 1});
  SmallVector<int64_t, 4> candidateTileSizes;
  candidateTileSizes.append({4 * cudaWarpSize, 2 * cudaWarpSize, cudaWarpSize});
  // Use the first tile size that can divide the shape. If the shape is not
  // aligned on any of the tile sizes pick the smallest tile of one element per
  // thread.
  int64_t lowerTs = cudaWarpSize;
  for (int64_t size : candidateTileSizes) {
    if (outputTypes[0].getShape().back() % size != 0) continue;
    lowerTs = size;
    break;
  }
  SmallVector<int64_t, 4> ts;
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  config.setTileSizes(op, ts, 0);  // Workgroup level.
  ts.back() = lowerTs / cudaWarpSize;
  config.setTileSizes(op, ts, 2);  // Thread level.
  return config;
}

static LaunchConfig getOpLaunchConfig(linalg::LinalgOp linalgOp) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation()))
    return getOpLaunchConfig(genericOp);
  return LaunchConfig();
}

namespace mlir {
namespace iree_compiler {

Optional<LaunchConfig> getCUDALaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig launchConfig;

  linalg::LinalgOp rootOperation;
  if (linalgOps.empty()) return llvm::None;
  if (linalgOps.size() == 1) rootOperation = *linalgOps.begin();
  // if there is more than one linalg op, look for the root one.
  for (linalg::LinalgOp op : linalgOps) {
    if (isa<linalg::BatchMatmulOp, linalg::MatmulOp>(op.getOperation())) {
      rootOperation = op;
      break;
    }
  }
  if (!rootOperation) {
    // No root operations found. Dont need to do anything.
    return llvm::None;
  }
  launchConfig = getOpLaunchConfig(rootOperation);
  launchConfig.setRootOperation(rootOperation.getOperation());

  if (failed(propogateRootOperationLaunchConfig(launchConfig, rootOperation,
                                                dependenceGraph)))
    return llvm::None;
  return launchConfig;
}

}  // namespace iree_compiler
}  // namespace mlir
