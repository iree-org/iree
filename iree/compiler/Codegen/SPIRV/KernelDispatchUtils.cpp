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

#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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

namespace {
struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};
}  // namespace

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
static LogicalResult setMaliSpecificConfig(FuncOp entryPoint,
                                           const spirv::TargetEnv &targetEnv,
                                           linalg::BatchMatmulOp op) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);
  // If the shape size is unknonw fall back to none vectorized path.
  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return failure();
  }

  // Get a vector of best tile size ordered from best to worst.
  SmallVector<TileWorkgroupSizePair, 4> workgroupLevelTs;
  int64_t dstSize = lhsShape[0] * lhsShape[1] * rhsShape[2];
  getMaliBestMatMulTileSizes(
      op.inputs()[0].getType().cast<ShapedType>().getElementType(),
      workgroupLevelTs, dstSize);
  for (TileWorkgroupSizePair pair : workgroupLevelTs) {
    if (lhsShape[1] % pair.tileSize[0] != 0 ||
        rhsShape[2] % pair.tileSize[1] != 0 ||
        lhsShape[2] % pair.tileSize[2] != 0) {
      continue;
    }

    SmallVector<int64_t, 4> batchTs;
    batchTs.append({1, pair.tileSize[0], pair.tileSize[1], pair.tileSize[2]});
    TileSizesListType tileSizes;
    tileSizes.emplace_back(batchTs);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    SmallVector<int64_t, 4> invocationLevelTs = {
        batchTs[0], batchTs[1] / pair.workgroupSize[1],
        batchTs[2] / pair.workgroupSize[0], batchTs[3]};
    tileSizes.emplace_back(invocationLevelTs);
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes, /*nativeVectorSize=*/ArrayRef<int64_t>{},
        IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize,
        pair.workgroupSize);
  }
  return failure();
}

/// Launch config for `linalg.batchmatmul`.
static LogicalResult setRootConfig(FuncOp entryPoint,
                                   const spirv::TargetEnv &targetEnv,
                                   linalg::BatchMatmulOp op) {
  if (succeeded(setMaliSpecificConfig(entryPoint, targetEnv, op))) {
    return success();
  }
  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  // This is just being hard-wired for now to be minimal viable, but this can be
  // decided better when we have better estimates of device charecteristics.
  const int64_t nRowsPerWorkitem = 1;
  const int64_t nColsPerWorkitem = 1;
  const int64_t nBatchesPerWorkitem = 1;
  int64_t tileSizeK = 0;
  SmallVector<int64_t, 4> workgroupLevel = {
      nBatchesPerWorkitem, nRowsPerWorkitem * workgroupSize[1],
      nColsPerWorkitem * workgroupSize[0], tileSizeK};
  SmallVector<int64_t, 4> invocationLevel = {
      nBatchesPerWorkitem, nRowsPerWorkitem, nColsPerWorkitem, 0};

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupLevel));
  tileSizes.emplace_back();  // subgroup level
  tileSizes.emplace_back(std::move(invocationLevel));
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSize=*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute, workgroupSize);
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
static LogicalResult setConfigForCooperativeMatmul(
    FuncOp entryPoint, const spirv::TargetEnv &targetEnv, linalg::MatmulOp op) {
  if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
      !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
    return failure();

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);
  // If the shape size is unknonw fall back to none vectorized path.
  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return failure();
  }

  auto resourceLimits = targetEnv.getResourceLimits();
  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };
  auto outputElementType = getElementType(op.outputs()[0]);
  Optional<SmallVector<int64_t, 4>> coopMatmulSize =
      getCooperativeMatmulSubgroupSize(
          resourceLimits, getElementType(op.inputs()[0]),
          getElementType(op.inputs()[1]), outputElementType, outputElementType);
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

  // For now this is being hard-wired to be {4, 4, 2}. This can actually be set
  // to whatever, but ultimately depends on register pressure.
  const int64_t numVecMatmulPerSubgroupX = 4;
  const int64_t numVecMatmulPerSubgroupY = 4;
  const int64_t numVecMatmulPerSubgroupK = 2;
  SmallVector<int64_t, 4> ts = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1],
      numVecMatmulPerSubgroupK * (*coopMatmulSize)[2]};
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(ts));

  int64_t subgroupSize =
      resourceLimits.subgroup_size().getValue().getSExtValue();
  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
  SmallVector<int64_t, 4> subgroupTs = {
      numVecMatmulPerSubgroupY * (*coopMatmulSize)[0],
      numVecMatmulPerSubgroupX * (*coopMatmulSize)[1]};
  tileSizes.emplace_back(std::move(subgroupTs));
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSize=*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize, workgroupSize);
}

/// Launch config for element-wise linalg.generic.
LogicalResult setDefaultRootConfig(FuncOp entryPoint,
                                   const spirv::TargetEnv &targetEnv,
                                   Operation *op) {
  auto partitionedLoops = getPartitionedLoops(op);
  if (partitionedLoops.empty()) {
    // Serialized computation.
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, /*tileSizes =*/TileSizesListType{{}},
        /*nativeVectorSize=*/ArrayRef<int64_t>{},
        IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize, {1, 1, 1});
  }

  // Skip vectorization for non-minor identity inputs as it generates
  // transfer_read ops with permutation maps that we currently cannot lower.
  // TODO: Remove this restriction once the lowering of the permutation map is
  // supported in core.
  int64_t subgroupSize =
      targetEnv.getResourceLimits().subgroup_size().getValue().getSExtValue();

  int64_t lowerWorkgroupTs = subgroupSize;
  int64_t lowerThreadTs = 1;
  IREE::HAL::DispatchLoweringPassPipeline pipeline =
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool vectorize = false;
    // TODO(thomasraoux): Lowering of integers other than i32 may require
    // emulation. This is currently not supported for vector operation.
    // Re-enable this when the bug is fixed on SPIR-V lowering side.
    auto outputShape = getUntiledResultShape(linalgOp, 0);
    if (!linalgOp.hasIndexSemantics() &&
        llvm::all_of(linalgOp.getIndexingMaps(),
                     [](AffineMap &map) { return map.isMinorIdentity(); }) &&
        llvm::all_of(
            linalgOp->getOperands(),
            [](Value operand) {
              auto shapedType = operand.getType().dyn_cast<ShapedType>();
              Type elementType = (shapedType ? shapedType.getElementType()
                                             : operand.getType());
              return elementType.isa<FloatType>() || elementType.isInteger(32);
            }) &&
        llvm::all_of(outputShape,
                     [](int64_t dim) { return !ShapedType::isDynamic(dim); })) {
      vectorize = true;
    }
    SmallVector<int64_t, 4> candidateTileSizes;
    if (vectorize) {
      candidateTileSizes.append({4 * subgroupSize, 2 * subgroupSize});
    }
    candidateTileSizes.push_back(subgroupSize);
    for (int64_t size : candidateTileSizes) {
      if (outputShape.back() % size != 0) continue;
      lowerWorkgroupTs = size;
      break;
    }
    if (lowerWorkgroupTs <= subgroupSize ||
        outputShape.back() % lowerWorkgroupTs != 0) {
      vectorize = false;
    }
    if (vectorize) {
      lowerThreadTs = lowerWorkgroupTs / subgroupSize;
      pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
    }
  }

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
  int64_t lowerTs = workgroupSize[0];
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t, 4> workgroupTileSize(loopDepth, 1),
      threadTileSize(loopDepth, 1);
  workgroupTileSize.back() = lowerWorkgroupTs;
  threadTileSize.back() = lowerThreadTs;
  TileSizesListType tileSizes;
  tileSizes.emplace_back(workgroupTileSize);  // Workgroup level
  tileSizes.emplace_back();                   // Subgroup level
  tileSizes.emplace_back(threadTileSize);     // Invocation level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      /*nativeVectorSize =*/ArrayRef<int64_t>{}, pipeline, workgroupSize);
}

/// Launch configuration for different known GPU configuration.
static LogicalResult setTargetSpecificConfig(FuncOp entryPoint,
                                             const spirv::TargetEnv &targetEnv,
                                             linalg::MatmulOp op) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);
  // If the shape size is unknonw fall back to none vectorized path.
  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return failure();
  }

  // Pick ideal tile size based on the type.
  SmallVector<TileWorkgroupSizePair, 4> workgroupLevelTs;
  int64_t dstSize = lhsShape[0] * rhsShape[1];
  getMaliBestMatMulTileSizes(
      op.inputs()[0].getType().cast<ShapedType>().getElementType(),
      workgroupLevelTs, dstSize);
  for (TileWorkgroupSizePair pair : workgroupLevelTs) {
    if (lhsShape[0] % pair.tileSize[0] != 0 ||
        rhsShape[1] % pair.tileSize[1] != 0 ||
        lhsShape[1] % pair.tileSize[2] != 0) {
      continue;
    }

    TileSizesListType tileSizes;
    SmallVector<int64_t, 4> matmulTS(pair.tileSize.begin(),
                                     pair.tileSize.end());
    tileSizes.emplace_back(matmulTS);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    SmallVector<int64_t, 4> invocationLevelTs = {
        matmulTS[0] / pair.workgroupSize[1],
        matmulTS[1] / pair.workgroupSize[0], matmulTS[2]};
    tileSizes.emplace_back(invocationLevelTs);
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes,
        /*nativeVectorSize =*/ArrayRef<int64_t>{},
        IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize,
        pair.workgroupSize);
  }
  return failure();
}

LogicalResult setRootConfig(FuncOp entryPoint,
                            const spirv::TargetEnv &targetEnv,
                            linalg::MatmulOp op) {
  if (succeeded(setConfigForCooperativeMatmul(entryPoint, targetEnv, op))) {
    return success();
  }
  if (succeeded(setTargetSpecificConfig(entryPoint, targetEnv, op))) {
    return success();
  }

  unsigned maxWorkgroupSize = targetEnv.getResourceLimits()
                                  .max_compute_workgroup_invocations()
                                  .getInt();
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);
  const int nRowsPerWorkitem = 1;
  const int nColsPerWorkitem = 1;
  int64_t tileSizeK = 0;

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  int64_t M = lhsShape[0];
  int64_t N = rhsShape[1];
  int64_t K = lhsShape[1];

  SmallVector<int64_t, 4> workgroupLevel = {
      getMinIfShapeStatic(M, nRowsPerWorkitem * workgroupSize[1]),
      getMinIfShapeStatic(N, nColsPerWorkitem * workgroupSize[0]),
      getMinIfShapeStatic(K, tileSizeK)};
  SmallVector<int64_t, 4> invocationLevel = {1, 1, 0};

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupLevel));
  tileSizes.emplace_back();  // subgroup level
  tileSizes.emplace_back(std::move(invocationLevel));
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes, /*nativeVectorSize =*/ArrayRef<int64_t>{},
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute, workgroupSize);
}

static LogicalResult setMaliSpecificConfig(
    FuncOp entryFn, linalg::ConvInputNHWCFilterHWCFOp op) {
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape =
      getUntiledResultShape(cast<linalg::LinalgOp>(op.getOperation()), 0);
  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return failure();
  }

  bool isInputTilable = inputShape[3] % 4 == 0 || inputShape[3] < 4;
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

    bool isOutputTilable = (outputShape[0] == 1) &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    TileSizesListType tileSizes;
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

    if (failed(setOpConfigAndEntryPointFnTranslation(
            entryFn, op, tileSizes, /*nativeVectorSize=*/ArrayRef<int64_t>{},
            IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize,
            workgroupSize)))
      return failure();

    // Let the entry point region to return fully static number of workgroups.
    // This is needed for folding `affine.min` ops to expose static-shaped tiled
    // convolution for vectorization.
    // TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
    auto numWorkgroupsFn = [&](OpBuilder &b, Location loc,
                               std::array<Value, 3>) {
      std::array<Value, 3> xyz;
      for (unsigned i = 0; i < 3; ++i) {
        int64_t count = outputShape[i + 1] / tileSize[i];
        xyz[2 - i] = b.create<ConstantIndexOp>(loc, count);
      }
      return xyz;
    };

    OpBuilder builder(op.getContext());
    return defineWorkgroupCountRegion(builder, entryFn, numWorkgroupsFn);
  }
  return failure();
}

LogicalResult setRootConfig(FuncOp entryPoint,
                            const spirv::TargetEnv &targetEnv,
                            linalg::ConvInputNHWCFilterHWCFOp op) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM &&
      succeeded(setMaliSpecificConfig(entryPoint, op))) {
    return success();
  }
  return success();
}

static LogicalResult setMaliSpecificConfig(
    FuncOp entryFn, linalg::DepthwiseConvInputNHWCFilterHWCOp op) {
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape =
      getUntiledResultShape(cast<linalg::LinalgOp>(op.getOperation()), 0);
  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return failure();
  }

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

    bool isOutputTilable = outputShape[0] == 1 &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    SmallVector<int64_t, 4> workgroupLevel = {/*batch=*/0,
                                              /*output_height=*/tileSize[0],
                                              /*output_width=*/tileSize[1],
                                              /*output_channel=*/tileSize[2]};
    TileSizesListType tileSizes;
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

    if (failed(setOpConfigAndEntryPointFnTranslation(
            entryFn, op, tileSizes, /*nativeVectorSize=*/ArrayRef<int64_t>{},
            IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize,
            workgroupSize)))
      return failure();

    // Let the entry point region to return fully static number of workgroups.
    // This is needed for folding `affine.min` ops to expose static-shaped tiled
    // convolution for vectorization.
    // TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
    auto numWorkgroupsFn = [&](OpBuilder &b, Location loc,
                               std::array<Value, 3>) {
      std::array<Value, 3> xyz;
      for (unsigned i = 0; i < 3; ++i) {
        int64_t count = outputShape[i + 1] / tileSize[i];
        xyz[2 - i] = b.create<ConstantIndexOp>(loc, count);
      }
      return xyz;
    };

    OpBuilder builder(op.getContext());
    return defineWorkgroupCountRegion(builder, entryFn, numWorkgroupsFn);
  }
  return failure();
}

static LogicalResult setRootConfig(
    FuncOp entryPoint, const spirv::TargetEnv &targetEnv,
    linalg::DepthwiseConvInputNHWCFilterHWCOp op) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM &&
      succeeded(setMaliSpecificConfig(entryPoint, op))) {
    return success();
  }
  return success();
}

/// Helper function to generate the number of workgroups when the
/// `SPIRVDistributeToGlobalID` is used.
// TODO(ravishankarm): Remove this when that pipeline is deprecated.
static LogicalResult setTranslationUsingDistributeToGlobalId(
    FuncOp funcOp, ArrayRef<int64_t> workgroupSize) {
  auto entryPointOp = getEntryPoint(funcOp);
  MLIRContext *context = entryPointOp.getContext();
  auto translationInfo = buildTranslationInfo(
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistributeToGlobalID,
      /*workloadPerWorkgroup =*/{}, context);
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
  OpBuilder builder(context);
  int64_t workgroupSizeX = workgroupSize[0];
  auto numWorkgroupsFn =
      [workgroupSizeX](OpBuilder &b, Location loc,
                       std::array<Value, 3> workload) -> std::array<Value, 3> {
    AffineExpr e1, e2, e3;
    bindSymbols(b.getContext(), e1, e2, e3);
    AffineExpr expr = e1 * e2 * e3;
    expr = expr.ceilDiv(workgroupSizeX);
    Value numWorkgroupsX = linalg::applyMapToValues(
        b, loc, AffineMap::get(0, 3, expr), workload)[0];
    Value one = b.create<ConstantIndexOp>(loc, 1);
    return {numWorkgroupsX, one, one};
  };
  return defineWorkgroupCountRegion(builder, funcOp, numWorkgroupsFn);
}

LogicalResult initSPIRVLaunchConfig(ModuleOp module) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(module);

  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    if (getTranslationInfo(entryPointOp)) continue;
    SmallVector<Operation *, 4> computeOps;
    SmallVector<Operation *, 4> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }
    spirv::TargetEnv targetEnv(spirv::lookupTargetEnv(funcOp));
    int64_t subgroupSize =
        targetEnv.getResourceLimits().subgroup_size().getValue().getSExtValue();

    if (computeOps.empty()) {
      // TODO(ravishankarm): `tensor.insert_slice` is not a compute op but still
      // needs to be handled in dispatch region. For now it is handled in
      // ConvertToGPU pass. Eventually this will be handled as a compute
      // op. This is just to keep scope of change to dynamic pass pipelines
      // limited. Remove this when dropping ConvertToGPU pass.
      if (failed(getFilteredOps(
              funcOp,
              [](Operation *op) {
                return isa<tensor::InsertSliceOp, tensor::ExtractSliceOp>(op);
              },
              computeOps, tiledLoops)) ||
          computeOps.empty()) {
        continue;
      }
      std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
      if (failed(
              setTranslationUsingDistributeToGlobalId(funcOp, workgroupSize))) {
        return computeOps[0]->emitOpError(
            "failed to set translation info for distributing to global IDs");
      }
      continue;
    }

    Operation *rootOperation = nullptr;
    for (Operation *computeOp : reverse(computeOps)) {
      if (!hasMarker(computeOp, getWorkgroupMarker())) continue;
      auto setConfigFn = [&](Operation *rootOp) -> LogicalResult {
        return TypeSwitch<Operation *, LogicalResult>(rootOp)
            .Case<linalg::BatchMatmulOp,
                  linalg::DepthwiseConvInputNHWCFilterHWCOp,
                  linalg::ConvInputNHWCFilterHWCFOp, linalg::MatmulOp>(
                [&](auto op) { return setRootConfig(funcOp, targetEnv, op); })
            .Default([&](Operation *) { return success(); });
      };
      if (failed(setConfigFn(computeOp))) {
        return failure();
      }
      // Check if the op configuration was set.
      if (getLoweringConfig(computeOp)) {
        if (rootOperation) {
          return computeOp->emitOpError(
              "unhandled multiple roots in dispatch region");
        }
        rootOperation = computeOp;
      }
    }

    // If there are still no roots, check for any generic op.
    if (!rootOperation) {
      for (Operation *computeOp : reverse(computeOps)) {
        if (!hasMarker(computeOp, getWorkgroupMarker())) continue;
        if (isa<linalg::FillOp, linalg::CopyOp>(computeOp)) continue;
        if (failed(setDefaultRootConfig(funcOp, targetEnv, computeOp))) {
          return failure();
        }
        if (getLoweringConfig(computeOp)) {
          if (rootOperation) {
            return computeOp->emitOpError(
                "unhandled multiple roots in dispatch region");
          }
          rootOperation = computeOp;
        }
      }
    }

    if (!rootOperation) {
      /// TODO(ravishankarm): This is setting the configuration for ops that are
      /// directly distributed to global invocation IDs. Remove this when
      /// SPIRVConvertToGPU is deprecated.
      for (Operation *computeOp : reverse(computeOps)) {
        if (hasMarker(computeOp, getWorkgroupMarker())) continue;
        if (isa<linalg::FillOp, linalg::CopyOp, linalg::GenericOp>(computeOp)) {
          std::array<int64_t, 3> workgroupSize = {1, 1, 1};
          auto linalgOp = cast<linalg::LinalgOp>(computeOp);
          if (getNumOuterParallelLoops(linalgOp)) {
            workgroupSize = {subgroupSize, 1, 1};
          }
          if (failed(setTranslationUsingDistributeToGlobalId(funcOp,
                                                             workgroupSize))) {
            return computeOp->emitOpError(
                "failed to set translation info for distributing to global "
                "IDs");
          }
          rootOperation = computeOp;
          break;
        }
      }
      if (rootOperation) continue;
    }

    // Propogate the configuration to the other ops.
    // TODO(ravishankarm, antiagainst): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    IREE::HAL::LoweringConfig config = getLoweringConfig(rootOperation);
    for (auto op : computeOps) {
      if (op == rootOperation) continue;
      setLoweringConfig(op, config);
    }
  }
  return success();
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
    // Unroll cooperative martrix load based on the size of the contract.
    VectorType dstVec;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract) return llvm::None;
      auto vecType = extract.getResult().getType().cast<VectorType>();
      if (dstVec && dstVec != vecType) return llvm::None;
      dstVec = vecType;
    }
    return SmallVector<int64_t, 4>(dstVec.getShape().begin(),
                                   dstVec.getShape().end());
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
    // Unroll cooperative martrix store based on the size of the contract.
    auto insert = op.vector().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insert) return llvm::None;
    ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
    return SmallVector<int64_t, 4>(shape.begin(), shape.end());
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
