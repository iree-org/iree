// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- KernelDispatchUtils.cpp - Utilities for generating dispatch info ---===//
//
// This file defines utility functions that can be used to get the information
// about tile sizes to use to partition work across workgroups, the workgroup
// sizes and to create information the dispatch on the host side needs to
// execute an entry point function (e.g. total number of workgroups).
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"

#include "iree/compiler/Conversion/LinalgToSPIRV/LaunchConfig.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Conversion/Utils/Utils.h"
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

/// Sets the `tileSizes` and `workgroupSize` for an Linalg `op` to the default,
/// where at most 3 inner parallel dimensions of `op` are tiled and distributed,
/// and each invocation handles one scalar elements.
// TODO(#5852): revisit the default here: they were chosen to get started and
// not very good.
static LogicalResult setDefaultTilingScheme(
    const spirv::TargetEnv &targetEnv, linalg::LinalgOp op,
    TileSizesListType &tileSizes, std::array<int64_t, 3> &workgroupSize) {
  auto maxWorkgroupSize =
      targetEnv.getResourceLimits().max_compute_workgroup_invocations();

  const int64_t tileSizeX = 32;
  const int64_t tileSizeY = maxWorkgroupSize.getInt() / tileSizeX;

  unsigned numParallelDims = getNumOuterParallelLoops(op);

  SmallVector<int64_t, 4> workgroupLevel(numParallelDims, 0);
  SmallVector<int64_t, 4> invocationLevel(numParallelDims, 0);

  if (numParallelDims >= 1) {
    workgroupLevel.back() = tileSizeX;
    invocationLevel.back() = 1;
  }
  if (numParallelDims >= 2) {
    workgroupLevel[numParallelDims - 2] = tileSizeY;
    invocationLevel[numParallelDims - 2] = 1;
  }
  if (numParallelDims >= 3) {
    workgroupLevel[numParallelDims - 3] = 1;
    invocationLevel[numParallelDims - 3] = 1;
  }

  tileSizes.emplace_back(std::move(workgroupLevel));
  tileSizes.emplace_back();  // Subgroup level
  tileSizes.emplace_back(std::move(invocationLevel));

  workgroupSize = {tileSizeX, tileSizeY, 1};

  return success();
}

/// Fills `inputTypes` and `outputTypes` with the original input/output types
/// for all tiles for `op`.
static std::tuple<SmallVector<ShapedType>, SmallVector<ShapedType>>
getInputOutputTypes(linalg::LinalgOp op) {
  SmallVector<ShapedType> inputTypes(op.getNumInputs()),
      outputTypes(op.getNumOutputs());
  auto inputOperands = op.getInputOperands();
  for (auto operand : enumerate(inputOperands)) {
    assert(!op.isScalar(operand.value()));
    inputTypes[operand.index()] =
        getUntiledType(operand.value()->get()).dyn_cast<ShapedType>();
  }
  auto outputOperands = op.getOutputOperands();
  for (auto operand : enumerate(outputOperands)) {
    outputTypes[operand.index()] =
        getUntiledType(operand.value()->get()).dyn_cast<ShapedType>();
  }
  return std::make_tuple(std::move(inputTypes), std::move(outputTypes));
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
  return setDefaultTilingScheme(targetEnv, op, tileSizes, config.workgroupSize);
}

static void getMaliBestMatMulTileSizes(
    Type elementType, SmallVectorImpl<TileWorkgroupSizePair> &tileSizes,
    int64_t dstSize) {
  if (elementType.isF16()) {
    const int64_t smallMatrixSizeThreshold = 512 * 512;
    // For smaller destination size we cannot fill out the GPU with bigger tile
    // sizes. Instead we pick smaller tiles along M and N to increase the number
    // of workgroups and a larger K tile size since we have lower pressure and
    // need extra instructions to hide latency.
    // TODO: The threshold needs to be fine tuned by doing exploration based on
    // matrix shapes.
    if (dstSize <= smallMatrixSizeThreshold) {
      tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 16}, {8, 2, 1}}));
    } else {
      tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {8, 2, 1}}));
      tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {8, 2, 1}}));
      tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 4}, {8, 2, 1}}));
    }
  } else {
    // TODO: Heuristic picked based on MobileNet performance. We need
    // auto-tuning to be able to make a smarter choice.
    const int64_t smallMatrixSizeThreshold = 20000;
    if (dstSize <= smallMatrixSizeThreshold) {
      tileSizes.push_back(TileWorkgroupSizePair({{4, 32, 16}, {8, 2, 1}}));
    }
    tileSizes.push_back(TileWorkgroupSizePair({{12, 32, 4}, {8, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{14, 32, 4}, {8, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{10, 32, 4}, {8, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{7, 64, 4}, {16, 1, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{8, 32, 4}, {8, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{6, 32, 4}, {8, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{24, 16, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 16, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{24, 8, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{40, 8, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{32, 8, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 8, 4}, {2, 8, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{1, 32, 16}, {8, 1, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{1, 32, 8}, {8, 1, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{1, 32, 4}, {8, 1, 1}}));
  }
}

/// Launch configuration for Mali GPU configuration.
static LogicalResult getMaliSpecificConfig(
    linalg::BatchMatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(op);

  ShapedType lhsType = inputTypes[0], rhsType = inputTypes[1];
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape())
    return failure();
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
  if (succeeded(getMaliSpecificConfig(op, targetEnv, options, tileSizes,
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
  SmallVector<int64_t, 4> workgroupLevel = {
      nBatchesPerWorkitem, nRowsPerWorkitem * config.workgroupSize[1],
      nColsPerWorkitem * config.workgroupSize[0], tileSizeK};
  SmallVector<int64_t, 4> invocationLevel = {
      nBatchesPerWorkitem, nRowsPerWorkitem, nColsPerWorkitem, 0};

  tileSizes.emplace_back(std::move(workgroupLevel));
  tileSizes.emplace_back();  // subgroup level
  tileSizes.emplace_back(std::move(invocationLevel));
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

  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(op);

  ShapedType lhsType = inputTypes[0], rhsType = inputTypes[1];
  ShapedType outputType = outputTypes[0];

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

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
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
LogicalResult getGenericOpLaunchConfig(linalg::LinalgOp linalgOp,
                                       const spirv::TargetEnv &targetEnv,
                                       const SPIRVCodegenOptions &options,
                                       TileSizesListType &tileSizes,
                                       LaunchConfigInfo &config) {
  // Skip vectorization for non-minor identity inputs as it generates
  // transfer_read ops with permutation maps that we currently cannot lower.
  // TODO: Remove this restriction once the lowering of the permutation map is
  // supported in core.
  bool vectorize = !linalgOp.hasIndexSemantics() &&
                   llvm::all_of(linalgOp.getIndexingMaps(), [](AffineMap &map) {
                     return map.isMinorIdentity();
                   });
  // TODO(thomasraoux): Lowering of integers other than i32 may require
  // emulation. This is currently not supported for vector operation. Re-enable
  // this when the bug is fixed on SPIR-V lowering side.
  if (llvm::any_of(linalgOp->getOperands(), [](Value operand) {
        Type memrefType = operand.getType().cast<MemRefType>().getElementType();
        return !memrefType.isa<FloatType>() && !memrefType.isInteger(32);
      }))
    vectorize = false;
  int64_t subgroupSize =
      targetEnv.getResourceLimits().subgroup_size().getValue().getSExtValue();
  config.workgroupSize[0] = subgroupSize;
  config.workgroupSize[1] = 1;
  config.workgroupSize[2] = 1;
  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(linalgOp);
  ShapedType outputShape = outputTypes[0];

  SmallVector<int64_t, 4> candidateTileSizes;
  // When Vectororization is not enabled we skil the second level of tiling and
  // fall back to convertToGPU which will map one element to one thread. To
  // avoid a mismatch in the number of workgroup dispatched, we pick a tile size
  // to have one element per thread.
  // TODO: Remove this once we switch to linalg on tensor path.
  if (vectorize) {
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
  unsigned numLoops = getNumOuterParallelLoops(linalgOp);
  SmallVector<int64_t, 4> ts(numLoops, 1);
  ts.back() = lowerTs;
  tileSizes.emplace_back(ts);  // Workgroup level
  tileSizes.emplace_back();    // Subgroup level

  if (!vectorize || outputShape.getShape().back() % lowerTs != 0) {
    ts.back() = 1;
    tileSizes.emplace_back(ts);  // Thread level
    config.vectorize = false;
  } else {
    ts.back() = lowerTs / subgroupSize;
    tileSizes.emplace_back(ts);  // Thread level
    // Vectorize only if we are processing more than one element per thread.
    config.vectorize = vectorize && (ts.back() > 1);
  }
  return success();
}

#define GET_GENERIC_OP_LAUNCH_CONFIG(opType)                            \
  template <>                                                           \
  LogicalResult getOpLaunchConfig(                                      \
      opType op, const spirv::TargetEnv &targetEnv,                     \
      const SPIRVCodegenOptions &options, TileSizesListType &tileSizes, \
      LaunchConfigInfo &config) {                                       \
    return getGenericOpLaunchConfig(op, targetEnv, options, tileSizes,  \
                                    config);                            \
  }

GET_GENERIC_OP_LAUNCH_CONFIG(linalg::GenericOp)

#undef GET_GENERIC_OP_LAUNCH_CONFIG

/// Launch configuration for different known GPU configuration.
static LogicalResult getTargetSpecificConfig(
    linalg::MatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(op);

  ShapedType lhsType = inputTypes[0], rhsType = inputTypes[1];
  // If the shape size is unknonw fall back to none vectorized path.
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape())
    return failure();

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
  if (succeeded(getConfigForCooperativeMatmul(op, targetEnv, options, tileSizes,
                                              config.workgroupSize,
                                              config.numSubgroups))) {
    config.vectorize = true;
    return success();
  }
  if (succeeded(getTargetSpecificConfig(op, targetEnv, options, tileSizes,
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
  const int nRowsPerWorkitem = 1;
  const int nColsPerWorkitem = 1;
  int64_t tileSizeK = 0;
  if (options.useWorkgroupMemory) {
    // TODO(#3131): This number should be decided based on the amount of shared
    // memory available (maybe). For now, just hard-wire it.
    tileSizeK = 32;
  }

  SmallVector<ShapedType> inputTypes;
  std::tie(inputTypes, std::ignore) = getInputOutputTypes(op);
  int64_t M = inputTypes[0].getShape()[0];
  int64_t N = inputTypes[1].getShape()[1];
  int64_t K = inputTypes[0].getShape()[1];

  SmallVector<int64_t, 4> workgroupLevel = {
      getMinIfShapeStatic(M, nRowsPerWorkitem * config.workgroupSize[1]),
      getMinIfShapeStatic(N, nColsPerWorkitem * config.workgroupSize[0]),
      getMinIfShapeStatic(K, tileSizeK)};
  SmallVector<int64_t, 4> invocationLevel = {1, 1, 0};

  tileSizes.emplace_back(std::move(workgroupLevel));
  tileSizes.emplace_back();  // subgroup level
  tileSizes.emplace_back(std::move(invocationLevel));
  return success();
}

static LogicalResult getMaliSpecificConfig(linalg::ConvInputNHWCFilterHWCFOp op,
                                           TileSizesListType &tileSizes,
                                           LaunchConfigInfo &config) {
  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(op);

  ShapedType inputType = inputTypes[0], outputType = outputTypes[0];
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape())
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

template <>
LogicalResult getOpLaunchConfig(linalg::ConvInputNHWCFilterHWCFOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM &&
      succeeded(getMaliSpecificConfig(op, tileSizes, config))) {
    return success();
  }

  return setDefaultTilingScheme(targetEnv, op, tileSizes, config.workgroupSize);
}

static LogicalResult getMaliSpecificConfig(
    linalg::DepthwiseConvInputNHWCFilterHWCOp op, TileSizesListType &tileSizes,
    LaunchConfigInfo &config) {
  SmallVector<ShapedType> inputTypes, outputTypes;
  std::tie(inputTypes, outputTypes) = getInputOutputTypes(op);

  ShapedType inputType = inputTypes[0], outputType = outputTypes[0];
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape())
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

  return setDefaultTilingScheme(targetEnv, op, tileSizes, config.workgroupSize);
}

Optional<LaunchConfig> initGPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    const SPIRVCodegenOptions &options, ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig launchConfig;
  if (!options.workgroupSize.empty()) {
    SmallVector<int64_t, 3> workgroupTileSizes(
        options.workgroupTileSizes.begin(), options.workgroupTileSizes.end());
    SmallVector<int64_t, 3> invocationTileSizes(
        options.invocationTileSizes.begin(), options.invocationTileSizes.end());
    for (linalg::LinalgOp linalgOp : linalgOps) {
      launchConfig.setTileSizes(linalgOp.getOperation(), workgroupTileSizes, 0);
      // Subgroup level.
      launchConfig.setTileSizes(linalgOp.getOperation(), {}, 1);
      // Invocation level.
      launchConfig.setTileSizes(linalgOp.getOperation(), invocationTileSizes,
                                2);
      launchConfig.setVectorize(true);
    }
    SmallVector<int64_t, 3> workgroupSize(options.workgroupSize.begin(),
                                          options.workgroupSize.end());
    launchConfig.setWorkgroupSize(workgroupSize);
  }

  if (linalgOps.empty()) return launchConfig;

  spirv::TargetEnv targetEnv(spirv::lookupTargetEnv(*linalgOps.begin()));

  Optional<linalg::LinalgOp> rootOperation = {};
  LaunchConfigInfo config;
#define DISPATCH(opName)                                                \
  if (auto op = dyn_cast<opName>(linalgOp.getOperation())) {            \
    rootOperation = linalgOp;                                           \
    if (launchConfig.hasTileSizes(linalgOp.getOperation())) break;      \
    TileSizesListType tileSizesInfo;                                    \
    if (failed(getOpLaunchConfig(op, targetEnv, options, tileSizesInfo, \
                                 config))) {                            \
      return llvm::None;                                                \
    }                                                                   \
    launchConfig.setTileSizes(op, tileSizesInfo);                       \
    break;                                                              \
  }

  for (linalg::LinalgOp linalgOp : linalgOps) {
    DISPATCH(linalg::BatchMatmulOp)
    DISPATCH(linalg::DepthwiseConvInputNHWCFilterHWCOp)
    DISPATCH(linalg::DepthwiseConvInputNHWCFilterHWCFOp)
    DISPATCH(linalg::ConvInputNWCFilterWCFOp)
    DISPATCH(linalg::ConvInputNHWCFilterHWCFOp)
    DISPATCH(linalg::ConvInputNDHWCFilterDHWCFOp)
    DISPATCH(linalg::MatmulOp)
    DISPATCH(linalg::PoolingNHWCMaxI8Op)
    DISPATCH(linalg::PoolingNHWCMaxI16Op)
    DISPATCH(linalg::PoolingNHWCMaxI32Op)
    DISPATCH(linalg::PoolingNHWCMaxFOp)
    DISPATCH(linalg::PoolingNHWCMinFOp)
    DISPATCH(linalg::PoolingNHWCSumFOp)
  }

  // Any generic operations found are made the root if no other op is the root
  if (!rootOperation) {
    for (linalg::LinalgOp linalgOp : reverse(linalgOps)) {
      size_t numLoops = getNumOuterParallelLoops(linalgOp);
      if (numLoops == 0 ||
          llvm::any_of(linalgOp.getIndexingMaps(), [](AffineMap &map) {
            return !map.isProjectedPermutation();
          })) {
        return llvm::None;
      }

      DISPATCH(linalg::GenericOp)
    }
  }

#undef DISPATCH

  if (!rootOperation) {
    return llvm::None;
  }

  launchConfig.setRootOperation(*rootOperation);
  if (options.workgroupSize.empty()) {
    launchConfig.setWorkgroupSize(config.workgroupSize);
    launchConfig.setVectorize(config.vectorize);
  }
  launchConfig.setNumSubgroups(config.numSubgroups);

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
  // Load 4 elements on the most inner dimension.
  for (auto dim : llvm::enumerate(op.permutation_map().getResults())) {
    if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() == op.permutation_map().getNumDims() - 1)
        nativeSize[dim.index()] = 4;
    }
  }
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
  // Store 4 elements on the most inner dimension.
  for (auto dim : llvm::enumerate(op.permutation_map().getResults())) {
    if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() == op.permutation_map().getNumDims() - 1)
        nativeSize[dim.index()] = 4;
    }
  }
  return nativeSize;
}

Optional<SmallVector<int64_t, 4>> getSPIRVNativeVectorSize(Operation *op) {
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
