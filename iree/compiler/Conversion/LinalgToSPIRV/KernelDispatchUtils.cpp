// Copyright 2020 Google LLC
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

//===- KernelDispatchUtils.cpp - Utilities for generating dispatch info ---===//
//
// This file defines utility functions that can be used to get the information
// about tile sizes to use to partition work across workgroups, the workgroup
// sizes and to create information the dispatch on the host side needs to
// execute an entry point function (e.g. total number of workgroups).
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "kernel-dispatch-utils"

namespace mlir {
namespace iree_compiler {

/// Given `nprocs` try to distribute it evenly across 2 logical x and y.
static std::tuple<int64_t, int64_t> distributeProcs2D(int64_t nprocs) {
  int64_t nprocs_x = std::max<int64_t>(
      1, static_cast<int64_t>(
             llvm::PowerOf2Ceil(static_cast<uint64_t>(std::sqrt(nprocs)))));
  return std::make_tuple(nprocs_x, nprocs / nprocs_x);
}

/// Returns the minimum of `shape` and `tileSize` if shape is static. If `shape`
/// is dynamic returns `tileSize`.
static int64_t getMinIfShapeStatic(int64_t shape, int64_t tileSize) {
  if (shape == ShapedType::kDynamicSize) return tileSize;
  return std::min(shape, tileSize);
}

/// Fills `inputTypes` and `outputTypes` with the original input/output types
/// for all tiles for `op`.
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

namespace {
struct LaunchConfigInfo {
  std::array<int64_t, 3> workgroupSize = {32, 1, 1};
  std::array<int64_t, 3> numSubgroups = {1, 1, 1};
  bool vectorize = false;
};

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};
}  // namespace

/// For a given operation `op`, compute the following configurations according
/// to SPIR-V `targetEnv` and `options`:
/// 1) number of tiling levels and tile sizes to use (updates `tileSizes`),
/// 2) workgroup size to use (updates `workgroupSize`),
/// 3) number of subgroups to use if two level tiling is used (updates
///    `numSubgroups`).
template <typename T>
static LogicalResult getOpLaunchConfig(T op, const spirv::TargetEnv &targetEnv,
                                       const SPIRVCodegenOptions &options,
                                       TileSizesListType &tileSizes,
                                       LaunchConfigInfo &config) {
  return op.emitError("undefined launch config for tiled operation");
}

static void getMaliBestMatMulTileSizes(
    Type elementType, SmallVectorImpl<TileWorkgroupSizePair> &tileSizes,
    int64_t dstSize) {
  const int64_t smallMatrixSizeThreshold = 512 * 512;
  if (elementType.isF16()) {
    // When the destination is smaller than the threshold, we prefer smaller
    // tiles to increase parallelism.
    // TODO: The threshold needs to be fine tuned by doing exploration based on
    // matrix shapes.
    if (dstSize <= smallMatrixSizeThreshold) {
      tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 8}, {8, 2, 1}}));
    } else {
      tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {8, 2, 1}}));
      tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {8, 2, 1}}));
      tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 4}, {8, 2, 1}}));
    }
  } else {
    tileSizes.push_back(TileWorkgroupSizePair({{8, 64, 4}, {16, 1, 1}}));
  }
}

/// Launch configuration for Mali GPU configuration.
static LogicalResult getMaliSpecificConfig(
    linalg::BatchMatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  SmallVector<ShapedType, 4> inputTypes, outputTypes;
  getInputOutputTypes(op, inputTypes, outputTypes);

  ShapedType lhsType = inputTypes[0], rhsType = inputTypes[1];
  assert(lhsType.getElementType() == rhsType.getElementType());

  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return failure();
  // Get a vector of best tile size ordered from best to worst.
  SmallVector<TileWorkgroupSizePair, 4> workgroupLevelTs;
  int64_t dstSize =
      lhsType.getDimSize(0) * lhsType.getDimSize(1) * rhsType.getDimSize(2);
  getMaliBestMatMulTileSizes(lhsType.getElementType(), workgroupLevelTs,
                             dstSize);
  for (TileWorkgroupSizePair pair : workgroupLevelTs) {
    if (lhsType.getDimSize(1) % pair.tileSize[0] != 0 ||
        rhsType.getDimSize(2) % pair.tileSize[1] != 0 ||
        lhsType.getDimSize(2) % pair.tileSize[2] != 0) {
      continue;
    }

    workgroupSize = pair.workgroupSize;
    SmallVector<int64_t, 4> batchTs;
    batchTs.append({1, pair.tileSize[0], pair.tileSize[1], pair.tileSize[2]});
    tileSizes.emplace_back(batchTs);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    SmallVector<int64_t, 4> invocationLevelTs = {
        batchTs[0], batchTs[1] / workgroupSize[1],
        batchTs[2] / workgroupSize[0], batchTs[3]};
    tileSizes.emplace_back(invocationLevelTs);
    return success();
  }
  return failure();
}

/// Launch config for `linalg.batchmatmul`.
template <>
LogicalResult getOpLaunchConfig(linalg::BatchMatmulOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  if (options.enableVectorization &&
      succeeded(getMaliSpecificConfig(op, targetEnv, options, tileSizes,
                                      config.workgroupSize,
                                      config.numSubgroups))) {
    config.vectorize = true;
    return success();
  }
  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  std::tie(config.workgroupSize[0], config.workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  config.workgroupSize[2] = 1;
  // This is just being hard-wired for now to be minimal viable, but this can be
  // decided better when we have better estimates of device charecteristics.
  const int64_t nRowsPerWorkitem = 1;
  const int64_t nColsPerWorkitem = 1;
  const int64_t nBatchesPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // This number should be decided based on the amount of shared memory
    // available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }
  assert(tileSizes.empty());
  SmallVector<int64_t, 4> ts = {
      nBatchesPerWorkitem, nRowsPerWorkitem * config.workgroupSize[1],
      nColsPerWorkitem * config.workgroupSize[0], tileSizeK};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

/// Returns the size of the co-operative matrix multiply operations on the
/// device.
static Optional<SmallVector<int64_t, 4>> getCooperativeMatmulSubgroupSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type initType, Type resultType) {
  for (auto coopMatmulProperties :
       resourceLimits.cooperative_matrix_properties_nv()
           .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>()) {
    if (coopMatmulProperties.a_type().getValue() == lhsType &&
        coopMatmulProperties.b_type().getValue() == rhsType &&
        coopMatmulProperties.c_type().getValue() == initType &&
        coopMatmulProperties.result_type().getValue() == resultType &&
        coopMatmulProperties.scope().getValue() == spirv::Scope::Subgroup) {
      return SmallVector<int64_t, 4>{
          coopMatmulProperties.m_size().getValue().getSExtValue(),
          coopMatmulProperties.n_size().getValue().getSExtValue(),
          coopMatmulProperties.k_size().getValue().getSExtValue()};
    }
  }
  return llvm::None;
}

/// Launch configuration for using spv.CooperativeMatrixMulAddNV
/// operations. Needs two levels of tiling.
static LogicalResult getConfigForCooperativeMatmul(
    linalg::MatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
    return failure();

  ShapedType lhsType = op.inputs().front().getType().cast<ShapedType>();
  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ShapedType rhsType = op.inputs().back().getType().cast<ShapedType>();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  ShapedType outputType = op.outputs().front().getType().cast<ShapedType>();

  auto resourceLimits = targetEnv.getResourceLimits();
  Optional<SmallVector<int64_t, 4>> coopMatmulSize =
      getCooperativeMatmulSubgroupSize(
          resourceLimits, lhsType.getElementType(), rhsType.getElementType(),
          outputType.getElementType(), outputType.getElementType());
  if (!coopMatmulSize) return failure();

  // Check that the matmul sizes are a multiple of the tilesize.
  auto isMultipleOf = [](int64_t s, int64_t ts) {
    return !ShapedType::isDynamic(s) && (s % ts) == 0;
  };
  if (!isMultipleOf(lhsShape[0], (*coopMatmulSize)[0]) ||
      !isMultipleOf(rhsShape[1], (*coopMatmulSize)[1]) ||
      !isMultipleOf(lhsShape[1], (*coopMatmulSize)[2]) ||
      !isMultipleOf(rhsShape[0], (*coopMatmulSize)[2]))
    return failure();

  if (options.useWorkgroupMemory) {
    numSubgroups[0] = 2;
    numSubgroups[1] = 2;
  } else {
    numSubgroups[0] = 1;
    numSubgroups[1] = 1;
  }
  numSubgroups[2] = 1;

  // For now this is being hard-wired to be {4, 4, 2}. This can actually be set
  // to whatever, but ultimately depends on register pressure.
  const int64_t numVecMatmulPerSubgroupX = 4;
  const int64_t numVecMatmulPerSubgroupY = 4;
  const int64_t numVecMatmulPerSubgroupK = 2;
  SmallVector<int64_t, 4> ts = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0] * numSubgroups[1],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1] * numSubgroups[0],
      numVecMatmulPerSubgroupK * (*coopMatmulSize)[2]};
  tileSizes.emplace_back(std::move(ts));

  int64_t subgroupSize =
      resourceLimits.subgroup_size().getValue().getSExtValue();
  workgroupSize[0] = numSubgroups[0] * numSubgroups[1] * subgroupSize;
  workgroupSize[1] = 1;
  workgroupSize[2] = 1;
  // Subgroup tile sizes
  SmallVector<int64_t, 4> subgroupTs = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1]};
  tileSizes.emplace_back(std::move(subgroupTs));
  return success();
}

/// Launch config for element-wise linalg.generic.
template <>
LogicalResult getOpLaunchConfig(linalg::GenericOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  int64_t subgroupSize =
      targetEnv.getResourceLimits().subgroup_size().getValue().getSExtValue();
  config.workgroupSize[0] = subgroupSize;
  config.workgroupSize[1] = 1;
  config.workgroupSize[2] = 1;
  ShapedType outputShape = op.getOutputShapedType(0);

  SmallVector<int64_t, 4> candidateTileSizes;
  // When Vectororization is not enabled we skil the second level of tiling and
  // fall back to convertToGPU which will map one element to one thread. To
  // avoid a mismatch in the number of workgroup dispatched, we pick a tile size
  // to have one element per thread.
  // TODO: Remove this once we switch to linalg on tensor path.
  if (options.enableVectorization) {
    candidateTileSizes.append({4 * subgroupSize, 2 * subgroupSize});
  }
  candidateTileSizes.push_back(subgroupSize);
  // Use the first tile size that can divide the shape. If the shape is not
  // aligned on any of the tile sizes pick the smallest tile of one element per
  // thread.
  int64_t lowerTs = config.workgroupSize[0];
  for (int64_t size : candidateTileSizes) {
    if (outputShape.getShape().back() % size != 0) continue;
    lowerTs = size;
    break;
  }
  SmallVector<int64_t, 4> ts;
  size_t numLoops = getNumOuterParallelLoops(op);
  ts.resize(numLoops, 1);
  ts.back() = lowerTs;
  tileSizes.emplace_back(ts);  // Workgroup level.
  tileSizes.emplace_back();    // Subgroup level.
  ts.back() = lowerTs / subgroupSize;
  tileSizes.emplace_back(ts);  // Thread level.
  // Vectorize only if we are processing more than one element per thread.
  config.vectorize = options.enableVectorization && (ts.back() > 1);
  return success();
}

/// Launch configuration for different known GPU configuration.
static LogicalResult getTargetSpecificConfig(
    linalg::MatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  SmallVector<ShapedType, 4> inputTypes, outputTypes;
  getInputOutputTypes(op, inputTypes, outputTypes);

  ShapedType lhsType = inputTypes[0], rhsType = inputTypes[1];
  assert(lhsType.getElementType() == rhsType.getElementType());

  // If the shape size is unknonw fall back to none vectorized path.
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return failure();
  // Pick ideal tile size based on the type.
  SmallVector<TileWorkgroupSizePair, 4> workgroupLevelTs;
  int64_t dstSize = lhsType.getDimSize(0) * rhsType.getDimSize(1);
  getMaliBestMatMulTileSizes(lhsType.getElementType(), workgroupLevelTs,
                             dstSize);
  for (TileWorkgroupSizePair pair : workgroupLevelTs) {
    if (lhsType.getDimSize(0) % pair.tileSize[0] != 0 ||
        rhsType.getDimSize(1) % pair.tileSize[1] != 0 ||
        lhsType.getDimSize(1) % pair.tileSize[2] != 0) {
      continue;
    }

    workgroupSize = pair.workgroupSize;
    SmallVector<int64_t, 4> matmulTS(pair.tileSize.begin(),
                                     pair.tileSize.end());
    tileSizes.emplace_back(matmulTS);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    SmallVector<int64_t, 4> invocationLevelTs = {matmulTS[0] / workgroupSize[1],
                                                 matmulTS[1] / workgroupSize[0],
                                                 matmulTS[2]};
    tileSizes.emplace_back(invocationLevelTs);
    return success();
  }
  return failure();
}

template <>
LogicalResult getOpLaunchConfig(linalg::MatmulOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  if (options.enableVectorization &&
      succeeded(getConfigForCooperativeMatmul(op, targetEnv, options, tileSizes,
                                              config.workgroupSize,
                                              config.numSubgroups))) {
    config.vectorize = true;
    return success();
  } else if (options.enableVectorization &&
             succeeded(getTargetSpecificConfig(op, targetEnv, options,
                                               tileSizes, config.workgroupSize,
                                               config.numSubgroups))) {
    config.vectorize = true;
    return success();
  }

  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  std::tie(config.workgroupSize[0], config.workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  config.workgroupSize[2] = 1;
  const int nRowsPerWorkitem = 1;
  const int nColsPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // TODO(#3131): This number should be decided based on the amount of shared
    // memory available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }
  assert(tileSizes.empty());
  int64_t M = op.inputs()[0].getType().cast<ShapedType>().getShape()[0];
  int64_t N = op.inputs()[1].getType().cast<ShapedType>().getShape()[1];
  int64_t K = op.inputs()[0].getType().cast<ShapedType>().getShape()[1];
  SmallVector<int64_t, 4> ts = {
      getMinIfShapeStatic(M, nRowsPerWorkitem * config.workgroupSize[1]),
      getMinIfShapeStatic(N, nColsPerWorkitem * config.workgroupSize[0]),
      getMinIfShapeStatic(K, tileSizeK)};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

template <typename ConvOpTy>
static LogicalResult getMaliSpecificConfig(ConvOpTy op,
                                           TileSizesListType &tileSizes,
                                           LaunchConfigInfo &config) {
  Operation *operation = op.getOperation();
  if (!isa<linalg::ConvInputNHWCFilterHWCFOp>(operation)) return failure();

  SmallVector<ShapedType, 4> inputTypes, outputTypes;
  getInputOutputTypes(op, inputTypes, outputTypes);

  ShapedType inputType = inputTypes[0], outputType = outputTypes[0];
  if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
    return failure();

  bool isInputTilable =
      inputType.getDimSize(3) % 4 == 0 || inputType.getDimSize(3) < 4;
  if (!isInputTilable) return failure();

  // A list of preferred tile sizes and workgroup sizes. This is for Mali
  // G77 now and it's fairly ad-hoc. We need to have a better story for
  // incorporating such information.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{4, 4, 16}, {4, 4, 1}},
      {{2, 2, 64}, {16, 1, 1}},
      {{4, 8, 8}, {2, 4, 2}},
      {{2, 2, 32}, {8, 2, 1}},
      {{1, 1, 32}, {8, 1, 1}}};

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    auto outputShape = outputType.getShape();
    bool isOutputTilable = (outputShape[0] == 1) &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    SmallVector<int64_t, 4> workgroupLevel = {
        /*batch=*/0, /*output_height=*/tileSize[0],
        /*output_width=*/tileSize[1], /*output_channel=*/tileSize[2]};
    tileSizes.emplace_back(std::move(workgroupLevel));

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization  or shared memory.
    tileSizes.emplace_back();

    SmallVector<int64_t, 4> invocationLevel = {
        /*batch=*/0, /*output_height=*/tileSize[0] / workgroupSize[2],
        /*output_width=*/tileSize[1] / workgroupSize[1],
        /*output_channel=*/tileSize[2] / workgroupSize[0]};
    tileSizes.emplace_back(invocationLevel);

    // Finally, for each invocation, we use tiling to generate loops to loop
    // over the filter's height (step 1), width (step 1), and input channel
    // (step 4) dimensions.
    SmallVector<int64_t, 4> fourthLevel = {0, 0, 0, 0, 1, 1, 4};
    tileSizes.emplace_back(fourthLevel);

    config.workgroupSize = workgroupSize;
    config.vectorize = true;

    return success();
  }

  return failure();
}

template <typename T>
LogicalResult getConvOpLaunchConfig(T op, const spirv::TargetEnv &targetEnv,
                                    const SPIRVCodegenOptions &options,
                                    TileSizesListType &tileSizes,
                                    LaunchConfigInfo &config) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM &&
      succeeded(getMaliSpecificConfig(op, tileSizes, config))) {
    return success();
  }

  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  SmallVector<int64_t, 4> ts;
  if (options.usingLinalgOnTensors) {
    ts.assign({0, 1, tileSizeY, tileSizeX});
  } else {
    ts.assign({1, tileSizeY, tileSizeX});
  }
  tileSizes.emplace_back(std::move(ts));
  config.workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

#define GET_CONV_LAUNCH_CONFIG(opType)                                       \
  template <>                                                                \
  LogicalResult getOpLaunchConfig(                                           \
      opType op, const spirv::TargetEnv &targetEnv,                          \
      const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,      \
      LaunchConfigInfo &config) {                                            \
    return getConvOpLaunchConfig(op, targetEnv, options, tileSizes, config); \
  }

GET_CONV_LAUNCH_CONFIG(linalg::ConvInputNWCFilterWCFOp)
GET_CONV_LAUNCH_CONFIG(linalg::ConvInputNHWCFilterHWCFOp)
GET_CONV_LAUNCH_CONFIG(linalg::ConvInputNDHWCFilterDHWCFOp)

#undef GET_CONV_LAUNCH_CONFIG

static LogicalResult getMaliSpecificConfig(
    linalg::DepthwiseConvInputNHWCFilterHWCOp op, TileSizesListType &tileSizes,
    LaunchConfigInfo &config) {
  SmallVector<ShapedType, 4> inputTypes, outputTypes;
  getInputOutputTypes(op, inputTypes, outputTypes);

  ShapedType inputType = inputTypes[0], outputType = outputTypes[0];
  if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
    return failure();

  // A list of preferred tile sizes and workgroup sizes. This is for Mali
  // G77 now and it's fairly ad-hoc. We need to have a better story for
  // incorporating such information.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{2, 2, 32}, {8, 2, 2}},
      {{1, 4, 16}, {4, 4, 1}},
      {{1, 1, 64}, {16, 1, 1}},
  };

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    auto outputShape = outputType.getShape();
    bool isOutputTilable = outputShape[0] == 1 &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    SmallVector<int64_t, 4> workgroupLevel = {/*batch=*/0,
                                              /*output_height=*/tileSize[0],
                                              /*output_width=*/tileSize[1],
                                              /*output_channel=*/tileSize[2]};
    tileSizes.emplace_back(std::move(workgroupLevel));

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization  or shared memory.
    tileSizes.emplace_back();

    SmallVector<int64_t, 4> invocationLevel = {
        /*batch=*/0, /*output_height=*/tileSize[0] / workgroupSize[2],
        /*output_width=*/tileSize[1] / workgroupSize[1],
        /*output_channel=*/tileSize[2] / workgroupSize[0]};
    tileSizes.emplace_back(invocationLevel);

    // Finally, for each invocation, we use tiling to generate loops to loop
    // over the filter's height (step 1) and width (step 1) dimensions.
    SmallVector<int64_t, 4> fourthLevel = {0, 0, 0, 0, 1, 1};
    tileSizes.emplace_back(fourthLevel);

    config.workgroupSize = workgroupSize;
    config.vectorize = true;

    return success();
  }
  return failure();
}

template <>
LogicalResult getOpLaunchConfig(linalg::DepthwiseConvInputNHWCFilterHWCOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM &&
      succeeded(getMaliSpecificConfig(op, tileSizes, config))) {
    return success();
  }

  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  SmallVector<int64_t, 4> ts;
  if (options.usingLinalgOnTensors) {
    ts.assign({0, 1, tileSizeY, tileSizeX});
  } else {
    ts.assign({1, tileSizeY, tileSizeX});
  }
  tileSizes.emplace_back(std::move(ts));
  config.workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

template <>
LogicalResult getOpLaunchConfig(linalg::DepthwiseConvInputNHWCFilterHWCFOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  SmallVector<int64_t, 4> ts = {1, tileSizeY, tileSizeX};
  tileSizes.emplace_back(std::move(ts));
  config.workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

template <typename PoolingOpTy>
static LogicalResult getPoolingOpLaunchConfig(
    PoolingOpTy op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    LaunchConfigInfo &config) {
  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  // Pooling op seems to be rank polymorphic but is not well specified enough to
  // be able to figure out which dimensions of the output correspond to the
  // pooled dimension and which are not. Need to fix that, but for now just use
  // a working heuristic.
  SmallVector<int64_t, 4> ts(std::min<int64_t>(
      op.getOutput(0).getType().template cast<ShapedType>().getRank(), 3));
  const int64_t tileSizeX = 32;
  int64_t tileSizeY = maxWorkgroupSize / tileSizeX;
  ts[ts.size() - 2] = tileSizeY;
  ts[ts.size() - 1] = tileSizeX;
  tileSizes.emplace_back(std::move(ts));
  config.workgroupSize = {tileSizeX, tileSizeY, 1};
  return success();
}

#define DEFINE_POOLING_OP_CONFIG(opName)                                \
  template <>                                                           \
  LogicalResult getOpLaunchConfig(                                      \
      opName op, const spirv::TargetEnv &targetEnv,                     \
      const SPIRVCodegenOptions &options, TileSizesListType &tileSizes, \
      LaunchConfigInfo &config) {                                       \
    return getPoolingOpLaunchConfig(op, targetEnv, options, tileSizes,  \
                                    config);                            \
  }

DEFINE_POOLING_OP_CONFIG(linalg::PoolingNHWCMaxOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingNHWCMinOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingNHWCSumOp)

#undef DEFINE_POOLINGOP_CONFIG

Optional<LaunchConfig> initGPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    const SPIRVCodegenOptions &options, ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig launchConfig;
  if (!options.workgroupSize.empty()) {
    SmallVector<int64_t, 3> tileSizes(options.tileSizes.begin(),
                                      options.tileSizes.end());
    for (linalg::LinalgOp linalgOp : linalgOps) {
      launchConfig.setTileSizes(linalgOp.getOperation(), tileSizes, 0);
    }
    SmallVector<int64_t, 3> workgroupSize(options.workgroupSize.begin(),
                                          options.workgroupSize.end());
    launchConfig.setWorkgroupSize(workgroupSize);
    return launchConfig;
  }

  if (linalgOps.empty()) return launchConfig;

  spirv::TargetEnv targetEnv(spirv::lookupTargetEnv(*linalgOps.begin()));

  Optional<linalg::LinalgOp> rootOperation = {};
  LaunchConfigInfo config;
  for (linalg::LinalgOp linalgOp : linalgOps) {
#define DISPATCH(opName)                                                     \
  if (auto op = dyn_cast<opName>(linalgOp.getOperation())) {                 \
    if (rootOperation) {                                                     \
      op.emitError("unhandled multiple root operations in dispatch region"); \
      return llvm::None;                                                     \
    }                                                                        \
    rootOperation = linalgOp;                                                \
    TileSizesListType tileSizesInfo;                                         \
    if (failed(getOpLaunchConfig(op, targetEnv, options, tileSizesInfo,      \
                                 config))) {                                 \
      return llvm::None;                                                     \
    }                                                                        \
    launchConfig.setTileSizes(op, tileSizesInfo);                            \
    launchConfig.setRootOperation(op);                                       \
    continue;                                                                \
  }

    DISPATCH(linalg::BatchMatmulOp)
    DISPATCH(linalg::DepthwiseConvInputNHWCFilterHWCOp)
    DISPATCH(linalg::DepthwiseConvInputNHWCFilterHWCFOp)
    DISPATCH(linalg::ConvInputNWCFilterWCFOp)
    DISPATCH(linalg::ConvInputNHWCFilterHWCFOp)
    DISPATCH(linalg::ConvInputNDHWCFilterDHWCFOp)
    DISPATCH(linalg::MatmulOp)
    DISPATCH(linalg::PoolingNHWCMaxOp)
    DISPATCH(linalg::PoolingNHWCMinOp)
    DISPATCH(linalg::PoolingNHWCSumOp)

#undef DISPATCH
  }

  if (!rootOperation) {
    for (linalg::LinalgOp linalgOp : linalgOps) {
      if (auto op = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
        if (getNumOuterParallelLoops(linalgOp) == 0 ||
            llvm::any_of(linalgOp.getIndexingMaps(), [](AffineMap &map) {
              return !map.isProjectedPermutation();
            })) {
          continue;
        }
        TileSizesListType tileSizesInfo;
        if (failed(getOpLaunchConfig(op, targetEnv, options, tileSizesInfo,
                                     config))) {
          continue;
        }
        launchConfig.setTileSizes(op, tileSizesInfo);
        launchConfig.setRootOperation(op);
        break;
      }
    }
  }

  launchConfig.setWorkgroupSize(config.workgroupSize);
  launchConfig.setNumSubgroups(config.numSubgroups);
  launchConfig.setVectorize(config.vectorize);

  if (!rootOperation) {
    // No root operations found. Dont need to do anything.
    return launchConfig;
  }

  if (failed(propogateRootOperationLaunchConfig(launchConfig, *rootOperation,
                                                dependenceGraph)))
    return llvm::None;

  // TODO(ravishankarm): Verify that the set configurations is within the device
  // limits.
  return launchConfig;
}

template <typename OpTy>
static Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize(OpTy op) {
  return llvm::None;
}

template <>
Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize<vector::ContractionOp>(
    vector::ContractionOp op) {
  auto targetEnvAttr = spirv::lookupTargetEnv(op);
  auto targetEnv = spirv::TargetEnv(targetEnvAttr);
  if (targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    return getCooperativeMatmulSubgroupSize(
        targetEnv.getResourceLimits(), op.getLhsType().getElementType(),
        op.getRhsType().getElementType(),
        op.getAccType().cast<VectorType>().getElementType(),
        op.getResultType().cast<VectorType>().getElementType());
  } else {
    unsigned lastParalleldim = 0;
    for (auto it : llvm::enumerate(op.iterator_types())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(op.iterator_types().size(), 1);
    nativeSize[lastParalleldim] = 4;
    // Map to vec4 fma operations.
    return nativeSize;
  }
}

template <>
Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize<vector::FMAOp>(
    vector::FMAOp op) {
  SmallVector<int64_t, 4> size(op.getType().getRank(), 1);
  size.back() = 4;
  return size;
}

template <>
Optional<SmallVector<int64_t, 4>> getOpNativeVectorSize<vector::TransferReadOp>(
    vector::TransferReadOp op) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    // Don't unroll cooperative martrix load as they should match the size of
    // the contract.
    return SmallVector<int64_t, 4>(op.getVectorType().getDimSize(0),
                                   op.getVectorType().getDimSize(1));
  }

  // Map to load4.
  auto rank = op.getVectorType().getRank();
  SmallVector<int64_t, 4> nativeSize(rank, 1);
  nativeSize.back() = 4;
  return nativeSize;
}

template <>
Optional<SmallVector<int64_t, 4>>
getOpNativeVectorSize<vector::TransferWriteOp>(vector::TransferWriteOp op) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(op));
  if (targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix)) {
    // Don't unroll cooperative martrix store as they should match the size of
    // the contract.
    return SmallVector<int64_t, 4>(op.getVectorType().getDimSize(0),
                                   op.getVectorType().getDimSize(1));
  }

  // Map to store4.
  auto rank = op.getVectorType().getRank();
  SmallVector<int64_t, 4> nativeSize(rank, 1);
  nativeSize.back() = 4;
  return nativeSize;
}

Optional<SmallVector<int64_t, 4>> getNativeVectorSize(Operation *op) {
#define DISPATCH(opname)                            \
  if (isa<opname>(op)) {                            \
    return getOpNativeVectorSize(cast<opname>(op)); \
  }

  DISPATCH(vector::ContractionOp)
  DISPATCH(vector::FMAOp)
  DISPATCH(vector::TransferReadOp)
  DISPATCH(vector::TransferWriteOp)

#undef DISPATCH

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Map elementwise ops to vec4.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank() - 1, 1);
      nativeSize.push_back(4);
      return nativeSize;
    }
  }
  return llvm::None;
}

}  // namespace iree_compiler
}  // namespace mlir
