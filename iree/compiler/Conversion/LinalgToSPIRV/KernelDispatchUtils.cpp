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
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "kernel-dispatch-utils"

namespace mlir {
namespace iree_compiler {

/// Name of the StrAttr that can be used to get the key to access the tile size
/// information.
static const char kLaunchInfoKey[] = "launch_info_key";

/// Given `nprocs` try to distribute it evenly across 2 logical x and y.
static std::tuple<int64_t, int64_t> distributeProcs2D(int64_t nprocs) {
  int64_t nprocs_x = std::max<int64_t>(
      1, static_cast<int64_t>(
             llvm::PowerOf2Ceil(static_cast<uint64_t>(std::sqrt(nprocs)))));
  return std::make_tuple(nprocs_x, nprocs / nprocs_x);
}

namespace {
struct LaunchConfigInfo {
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  // The corresponding indices of the loops that are distributed to workgroup
  // dimensions.
  std::array<int64_t, 3> workgroupLoopIndices = {0, 1, 2};
  std::array<int64_t, 3> numSubgroups = {1, 1, 1};
  bool vectorize = false;
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

static void getMaliBestMatMulTileSizes(Type elementType,
                                       SmallVectorImpl<int64_t> &tileSizes) {
  if (elementType.isF16()) {
    tileSizes.append({16, 64, 8});
  } else {
    tileSizes.append({8, 64, 4});
  }
}

/// Launch configuration for Mali GPU configuration.
static LogicalResult getMaliSpecificConfig(
    linalg::BatchMatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  auto lhsType = op.inputs()[0].getType().cast<MemRefType>();
  auto rhsType = op.inputs()[1].getType().cast<MemRefType>();
  assert(lhsType.getElementType() == rhsType.getElementType());
  // Pick ideal tile size based on the type.
  SmallVector<int64_t, 4> workgroupLevelTs(1, 1);
  getMaliBestMatMulTileSizes(lhsType.getElementType(), workgroupLevelTs);
  // Fall back to the none vectorize path for cases we don't handle.
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      lhsType.getDimSize(1) % workgroupLevelTs[1] != 0 ||
      rhsType.getDimSize(2) % workgroupLevelTs[2] != 0 ||
      lhsType.getDimSize(2) % workgroupLevelTs[3] != 0) {
    return failure();
  }

  workgroupSize[0] = targetEnv.getResourceLimits().subgroup_size().getInt();
  workgroupSize[1] = 1;
  workgroupSize[2] = 1;
  tileSizes.emplace_back(workgroupLevelTs);
  // No tiling at the subgroup level since this target doesn't use subgroup op
  // or shared memory.
  tileSizes.emplace_back();
  SmallVector<int64_t, 4> invocationLevelTs = {
      workgroupLevelTs[0], workgroupLevelTs[1],
      workgroupLevelTs[2] / workgroupSize[0], workgroupLevelTs[3]};
  tileSizes.emplace_back(invocationLevelTs);
  return success();
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
        spirv::symbolizeScope(
            coopMatmulProperties.scope().getValue().getZExtValue())
                .getValue() == spirv::Scope::Subgroup) {
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
  ShapedType outputType =
      op.output_buffers().front().getType().cast<ShapedType>();

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

/// Launch configuration for different known GPU configuration.
static LogicalResult getTargetSpecificConfig(
    linalg::MatmulOp op, const spirv::TargetEnv &targetEnv,
    const SPIRVCodegenOptions &options, TileSizesListType &tileSizes,
    std::array<int64_t, 3> &workgroupSize,
    std::array<int64_t, 3> &numSubgroups) {
  if (targetEnv.getVendorID() != spirv::Vendor::ARM) return failure();

  auto lhsType = op.inputs()[0].getType().cast<MemRefType>();
  auto rhsType = op.inputs()[1].getType().cast<MemRefType>();
  assert(lhsType.getElementType() == rhsType.getElementType());
  // Pick ideal tile size based on the type.
  SmallVector<int64_t, 4> workgroupLevelTs;
  getMaliBestMatMulTileSizes(lhsType.getElementType(), workgroupLevelTs);

  // Fall back to the none vectorize path for cases we don't handle.
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      lhsType.getDimSize(0) % workgroupLevelTs[0] != 0 ||
      rhsType.getDimSize(1) % workgroupLevelTs[1] != 0 ||
      lhsType.getDimSize(1) % workgroupLevelTs[2] != 0) {
    return failure();
  }

  workgroupSize[0] = targetEnv.getResourceLimits().subgroup_size().getInt();
  workgroupSize[1] = 1;
  workgroupSize[2] = 1;
  tileSizes.emplace_back(workgroupLevelTs);
  // No tiling at the subgroup level since this target doesn't use subgroup op
  // or shared memory.
  tileSizes.emplace_back();
  SmallVector<int64_t, 4> invocationLevelTs = {
      workgroupLevelTs[0], workgroupLevelTs[1] / workgroupSize[0],
      workgroupLevelTs[2]};
  tileSizes.emplace_back(invocationLevelTs);
  return success();
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
  SmallVector<int64_t, 4> ts = {nRowsPerWorkitem * config.workgroupSize[1],
                                nColsPerWorkitem * config.workgroupSize[0],
                                tileSizeK};
  tileSizes.emplace_back(std::move(ts));
  return success();
}

template <>
LogicalResult getOpLaunchConfig(linalg::ConvOp op,
                                const spirv::TargetEnv &targetEnv,
                                const SPIRVCodegenOptions &options,
                                TileSizesListType &tileSizes,
                                LaunchConfigInfo &config) {
  if (targetEnv.getVendorID() == spirv::Vendor::ARM) {
    auto inputType = op.getInput(1).getType().cast<MemRefType>();
    auto outputType = op.getOutputBufferType(0).cast<MemRefType>();
    if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
      const int tileWidth = 8;
      const int tileChannel = 32;

      auto outputShape = outputType.getShape();
      bool isInputTilable = inputType.getDimSize(3) % 4 == 0;
      bool isOutputTilable = outputShape[0] == 1 &&
                             outputShape[2] % tileWidth == 0 &&
                             outputShape[3] % tileChannel == 0;

      if (isInputTilable && isOutputTilable) {
        config.workgroupSize = {8, 2, 1};

        SmallVector<int64_t, 4> workgroupLevel = {
            /*batch=*/0, /*output_height=*/1,
            /*output_width=*/tileWidth,
            /*output_channel=*/tileChannel};
        tileSizes.emplace_back(std::move(workgroupLevel));

        // No tiling at the subgroup level given that we don't use subgroup
        // level syncrhonization  or shared memory.
        tileSizes.emplace_back();

        SmallVector<int64_t, 4> invocationLevel = {
            /*batch=*/0, /*output_height=*/1,
            /*output_width=*/tileWidth / config.workgroupSize[1],
            /*output_channel=*/tileChannel / config.workgroupSize[0]};
        tileSizes.emplace_back(invocationLevel);

        // We don't distribute along the batch dimension.
        config.workgroupLoopIndices = {1, 2, 3};
        config.vectorize = true;

        return success();
      }
    }
  }

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
      op.output().getType().template cast<ShapedType>().getRank(), 3));
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

DEFINE_POOLING_OP_CONFIG(linalg::PoolingMaxOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingMinOp)
DEFINE_POOLING_OP_CONFIG(linalg::PoolingSumOp)

#undef DEFINE_POOLINGOP_CONFIG

Optional<StringRef> LaunchConfig::getKey(Operation *op) const {
  StringAttr attr = op->getAttrOfType<StringAttr>(kLaunchInfoKey);
  if (!attr) return {};
  return attr.getValue();
}

LogicalResult LaunchConfig::init(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    const SPIRVCodegenOptions &options, ArrayRef<linalg::LinalgOp> linalgOps) {
  unsigned numTiledOps = 0;
  auto setKey = [&](Operation *op) -> std::string {
    std::string key = llvm::formatv("__op_num_{0}__", numTiledOps++).str();
    op->setAttr(Identifier::get(kLaunchInfoKey, context),
                StringAttr::get(key, context));
    return key;
  };

  if (!options.workgroupSize.empty()) {
    for (linalg::LinalgOp linalgOp : linalgOps)
      tileSizes[setKey(linalgOp)].emplace_back(options.tileSizes.begin(),
                                               options.tileSizes.end());
    workgroupSize = {1, 1, 1};
    for (unsigned i = 0,
                  e = std::min<unsigned>(3, options.workgroupSize.size());
         i != e; ++i)
      workgroupSize[i] = options.workgroupSize[i];
    return success();
  }

  if (linalgOps.empty()) return success();

  spirv::TargetEnv targetEnv(spirv::lookupTargetEnv(*linalgOps.begin()));

  Optional<linalg::LinalgOp> rootOperation = {};
  LaunchConfigInfo config;
  for (linalg::LinalgOp linalgOp : linalgOps) {
#define DISPATCH(opName)                                                  \
  if (auto op = dyn_cast<opName>(linalgOp.getOperation())) {              \
    if (rootOperation) {                                                  \
      return op.emitError(                                                \
          "unhandled multiple root operations in dispatch region");       \
    }                                                                     \
    rootOperation = linalgOp;                                             \
    TileSizesListType &tileSizesInfo = tileSizes[setKey(*rootOperation)]; \
    if (failed(getOpLaunchConfig(op, targetEnv, options, tileSizesInfo,   \
                                 config))) {                              \
      return failure();                                                   \
    }                                                                     \
    continue;                                                             \
  }

    DISPATCH(linalg::BatchMatmulOp)
    DISPATCH(linalg::ConvOp)
    DISPATCH(linalg::MatmulOp)
    DISPATCH(linalg::PoolingMaxOp)
    DISPATCH(linalg::PoolingMinOp)
    DISPATCH(linalg::PoolingSumOp)

#undef DISPATCH
  }
  workgroupSize = config.workgroupSize;
  workgroupLoopIndices = config.workgroupLoopIndices;
  numSubgroups = config.numSubgroups;
  vectorize = config.vectorize;
  if (!rootOperation) {
    // No root operations found. Dont need to do anything.
    return success();
  }

  // Check the dependencies going into and out of the root operation. For now
  // only the following dependencies are supported
  // - WAW dependencies going into the root operation.
  // - RAW dependencies going out of the root operation.
  // - WAW dependencies going out of the root operation.
  // i.e. there are no RAW dependences going into the root operation.
  auto inRAWDependencies = dependenceGraph.getDependencesInto(
      *rootOperation, linalg::LinalgDependenceGraph::RAW);
  if (!inRAWDependencies.empty()) {
    return rootOperation->getOperation()->emitError(
        "unhandled fusion of root operation with producer");
  }
  auto dependences =
      dependenceGraph.getDependentOperations(rootOperation.getValue());
  unsigned numOuterParallel = getNumOuterParallelLoops(*rootOperation);

  // Check that for all dependences into and out of the root operation,
  // - The result expressions of the indexing maps of the fused view in the
  //   producer and consumer must match for the parallel loops.
  for (auto dependence :
       dependenceGraph.getDependentOperations(rootOperation.getValue())) {
    unsigned viewIndex = dependence.indexingOpView.operandIndex;
    AffineMap indexingMap = rootOperation->getIndexingMap(viewIndex);
    linalg::LinalgOp fusedOp =
        cast<linalg::LinalgOp>(dependence.dependentOpView.op);
    unsigned fusedViewIndex = dependence.dependentOpView.operandIndex;
    AffineMap fusedIndexingMap = fusedOp.getIndexingMap(fusedViewIndex);
    if (indexingMap.getNumResults() < numOuterParallel ||
        fusedIndexingMap.getNumResults() < numOuterParallel ||
        !llvm::all_of(
            llvm::seq<unsigned>(0, numOuterParallel),
            [&indexingMap, fusedIndexingMap](unsigned i) {
              return indexingMap.getResult(i).isa<AffineDimExpr>() &&
                     fusedIndexingMap.getResult(i).isa<AffineDimExpr>() &&
                     indexingMap.getResult(i) == fusedIndexingMap.getResult(i);
            })) {
      return rootOperation->getOperation()->emitError(
          "unhandled fusion of root operation with all operations in the "
          "dispatch region");
    }
  }
  // The dependent operations get the same tile size information as the root
  // operation. To propogate that information, just use the same key as the root
  // operation.
  for (auto dependence : dependences) {
    dependence.dependentOpView.op->setAttr(
        Identifier::get(kLaunchInfoKey, context),
        StringAttr::get(getKey(*rootOperation).getValue(), context));
  }

  // TODO(ravishankarm): Verify that the set configurations is within the device
  // limits.
  return success();
}

void LaunchConfig::finalize(FuncOp funcOp) {
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    linalgOp.removeAttr(Identifier::get(kLaunchInfoKey, funcOp.getContext()));
  });
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
  DISPATCH(vector::TransferReadOp)
  DISPATCH(vector::TransferWriteOp)

#undef DISPATCH
  return llvm::None;
}

}  // namespace iree_compiler
}  // namespace mlir
