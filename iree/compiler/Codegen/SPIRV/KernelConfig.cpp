// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Given `nprocs`, tries to distribute it evenly across 2 logical dimensions.
std::tuple<int64_t, int64_t> distributeProcs2D(int64_t nprocs) {
  int64_t nprocs_x = std::max<int64_t>(
      1, static_cast<int64_t>(
             llvm::PowerOf2Ceil(static_cast<uint64_t>(std::sqrt(nprocs)))));
  return std::make_tuple(nprocs_x, nprocs / nprocs_x);
}

/// Returns the minimum of `shape` and `tileSize` if shape is static.
/// Returns `tileSize` otherwise.
int64_t getMinIfStaticShape(int64_t shape, int64_t tileSize) {
  if (shape == ShapedType::kDynamicSize) return tileSize;
  return std::min(shape, tileSize);
}

/// Defines the workgroup count region on entry point ops for the
/// `SPIRVDistributeToGlobalID` pipeline.
// TODO(ravishankarm): Remove this when that pipeline is deprecated.
LogicalResult setTranslationUsingDistributeToGlobalId(
    FuncOp funcOp, ArrayRef<int64_t> workgroupSize) {
  auto entryPointOp = getEntryPoint(funcOp);
  MLIRContext *context = entryPointOp.getContext();
  auto translationInfo = buildTranslationInfo(
      IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistributeToGlobalID,
      /*workloadPerWorkgroup =*/{}, context);
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
  OpBuilder builder(context);
  int64_t workgroupSizeX = workgroupSize[0];
  auto numWorkgroupsFn = [workgroupSizeX](OpBuilder &b, Location loc,
                                          std::array<Value, 3> workload) {
    AffineExpr e1, e2, e3;
    bindSymbols(b.getContext(), e1, e2, e3);
    AffineExpr expr = e1 * e2 * e3;
    expr = expr.ceilDiv(workgroupSizeX);
    Value numWorkgroupsX = linalg::applyMapToValues(
        b, loc, AffineMap::get(0, 3, expr), workload)[0];
    Value one = b.create<ConstantIndexOp>(loc, 1);
    return std::array<Value, 3>{numWorkgroupsX, one, one};
  };
  return defineWorkgroupCountRegion(builder, funcOp, numWorkgroupsFn);
}

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

Optional<detail::SPIRVCodeGenConfig> getOpConfig(
    spirv::ResourceLimitsAttr limits, linalg::BatchMatmulOp op) {
  unsigned maxWorkgroupSize =
      limits.max_compute_workgroup_invocations().getInt();

  // This is just being hard-wired for now to be minimal viable, but this can be
  // decided better when we have better estimates of device charecteristics.
  const int64_t numRowsPerThread = 1;
  const int64_t numColsPerThread = 1;
  const int64_t numBatchesPerThread = 1;
  const int64_t tileSizeK = 0;

  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);

  detail::SPIRVCodeGenConfig config = {};
  config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  config.workgroupTileSizes = {numBatchesPerThread,
                               numRowsPerThread * workgroupSize[1],
                               numColsPerThread * workgroupSize[0], tileSizeK};

  config.invocationTileSizes = {numBatchesPerThread, numRowsPerThread,
                                numColsPerThread, 0};

  config.workgroupSize = {1, 1, 1};
  std::tie(config.workgroupSize[0], config.workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);

  config.workgroupSize = workgroupSize;

  return config;
}

Optional<detail::SPIRVCodeGenConfig> getOpConfig(
    spirv::ResourceLimitsAttr limits, linalg::MatmulOp op) {
  unsigned maxWorkgroupSize =
      limits.max_compute_workgroup_invocations().getInt();

  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  std::tie(workgroupSize[0], workgroupSize[1]) =
      distributeProcs2D(maxWorkgroupSize);

  const int numRowsPerThread = 1;
  const int numColsPerThread = 1;
  int64_t tileSizeK = 0;

  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  int64_t M = lhsShape[0];
  int64_t N = rhsShape[1];
  int64_t K = lhsShape[1];

  detail::SPIRVCodeGenConfig config = {};
  config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  config.workgroupTileSizes = {
      getMinIfStaticShape(M, numRowsPerThread * workgroupSize[1]),
      getMinIfStaticShape(N, numColsPerThread * workgroupSize[0]),
      getMinIfStaticShape(K, tileSizeK)};

  config.invocationTileSizes = {1, 1, 0};

  config.workgroupSize = workgroupSize;

  return config;
}

//===----------------------------------------------------------------------===//
// Default Configuration
//===----------------------------------------------------------------------===//

Optional<detail::SPIRVCodeGenConfig> getDefaultOpConfig(
    spirv::ResourceLimitsAttr limits, Operation *op) {
  detail::SPIRVCodeGenConfig config = {};

  auto partitionedLoops = getPartitionedLoops(op);
  if (partitionedLoops.empty()) {
    config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
    config.workgroupSize = {1, 1, 1};
    return config;
  }

  const int64_t subgroupSize = limits.subgroup_size().getValue().getSExtValue();
  int64_t numElementsPerWorkgroup = subgroupSize;
  int64_t numElementsPerThread = 1;
  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  // Returns true if the given `operand` has 32-bit element type.
  auto has32BitElementType = [](Value operand) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    Type elementType =
        (shapedType ? shapedType.getElementType() : operand.getType());
    return elementType.isa<FloatType>() || elementType.isInteger(32);
  };

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool vectorize = false;
    auto outputShape = getUntiledResultShape(linalgOp, 0);

    if (!linalgOp.hasIndexSemantics() &&
        // Skip vectorization for non-minor identity inputs as it generates
        // vector.transfer_read ops with permutation maps that we currently
        // cannot lower.
        // TODO: Remove this restriction once the lowering of the permutation
        // map is supported in core.
        llvm::all_of(linalgOp.getIndexingMaps(),
                     [](AffineMap &map) { return map.isMinorIdentity(); }) &&
        // TODO(thomasraoux): Lowering of integers other than i32 may require
        // emulation. This is currently not supported for vector operation.
        // Re-enable this when the bug is fixed on SPIR-V lowering side.
        llvm::all_of(linalgOp->getOperands(), has32BitElementType) &&
        llvm::all_of(outputShape,
                     [](int64_t dim) { return !ShapedType::isDynamic(dim); })) {
      vectorize = true;
    }

    SmallVector<int64_t, 4> candidateTileSizes;
    if (vectorize) candidateTileSizes.push_back(4 * subgroupSize);
    candidateTileSizes.push_back(subgroupSize);

    for (int64_t size : candidateTileSizes) {
      if (outputShape.back() % size != 0) continue;
      numElementsPerWorkgroup = size;
      break;
    }

    if (numElementsPerWorkgroup <= subgroupSize ||
        outputShape.back() % numElementsPerWorkgroup != 0) {
      vectorize = false;
    }

    if (vectorize) {
      numElementsPerThread = numElementsPerWorkgroup / subgroupSize;
      pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
    }
  }

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t, 4> workgroupTileSize(loopDepth, 0);
  SmallVector<int64_t, 4> threadTileSize(loopDepth, 0);

  // Tiling along partitioned loops with size 1.
  for (int64_t loopIndex : partitionedLoops) {
    workgroupTileSize[loopIndex] = threadTileSize[loopIndex] = 1;
  }
  // Overwrite the configuration for the innermost dimension.
  workgroupTileSize.back() = numElementsPerWorkgroup;
  threadTileSize.back() = numElementsPerThread;

  config.pipeline = pipeline;
  config.workgroupTileSizes = workgroupTileSize;
  config.invocationTileSizes = threadTileSize;
  config.workgroupSize = workgroupSize;

  return config;
}

Optional<detail::SPIRVCodeGenConfig> getSPIRVOpConfig(
    const spirv::TargetEnv &targetEnv, Operation *rootOp) {
  Optional<detail::SPIRVCodeGenConfig> config;
  // First try to find a proper CodeGen configuration for the current
  // target architecture.
  switch (targetEnv.getVendorID()) {
    case spirv::Vendor::ARM:
      config = detail::getMaliCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::NVIDIA:
      config = detail::getNVIDIACodeGenConfig(targetEnv, rootOp);
      break;
    default:
      break;
  }
  if (config) return config;

  // Then try to use a default configuration.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  if (auto matmulOp = dyn_cast<linalg::BatchMatmulOp>(rootOp)) {
    config = getOpConfig(limits, matmulOp);
  } else if (auto matmulOp = dyn_cast<linalg::MatmulOp>(rootOp)) {
    config = getOpConfig(limits, matmulOp);
  } else if (isa<linalg::DepthwiseConvInputNHWCFilterHWCOp,
                 linalg::ConvInputNHWCFilterHWCFOp>(rootOp)) {
    config = getDefaultOpConfig(limits, rootOp);
  }

  return config;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

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
    spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
    int64_t subgroupSize = limits.subgroup_size().getValue().getSExtValue();

    // If the dispatch region does not contain tiled and distributed Linalg ops,
    // invoke the pipeline to distribute to global invocations.
    if (computeOps.empty() || llvm::none_of(computeOps, [](Operation *op) {
          return hasMarker(op, getWorkgroupMarker());
        })) {
      auto isInsertExtractSliceOp = [](Operation *op) {
        return isa<tensor::InsertSliceOp, tensor::ExtractSliceOp>(op);
      };
      // TODO(ravishankarm): `tensor.insert_slice` is not a compute op but still
      // needs to be handled in dispatch region. For now it is handled in
      // ConvertToGPU pass. Eventually this will be handled as a compute
      // op. This is just to keep scope of change to dynamic pass pipelines
      // limited. Remove this when dropping ConvertToGPU pass.
      if (failed(getFilteredOps(funcOp, isInsertExtractSliceOp, computeOps,
                                tiledLoops)) ||
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
    Optional<detail::SPIRVCodeGenConfig> spirvConfig;

    // Try to find a configuration according to a matmul/convolution op and use
    // it as the root op.
    for (Operation *computeOp : computeOps) {
      if (auto opConfig = getSPIRVOpConfig(targetEnv, computeOp)) {
        if (rootOperation) {
          return computeOp->emitOpError(
              "unhandled multiple roots in dispatch region");
        }
        rootOperation = computeOp;
        spirvConfig = opConfig;
      }
    }

    // If there are still no root op, check for any linalg.generic op.
    if (!rootOperation) {
      for (Operation *computeOp : computeOps) {
        if (isa<linalg::FillOp, linalg::CopyOp>(computeOp)) continue;

        if (auto opConfig = getDefaultOpConfig(limits, computeOp)) {
          if (rootOperation) {
            return computeOp->emitOpError(
                "unhandled multiple roots in dispatch region");
          }
          rootOperation = computeOp;
          spirvConfig = opConfig;
        }
      }
    }

    // Now compose and attach the `lowering.config` attribute to the root op.
    assert(rootOperation && "failed to discover root operation!");

    TileSizesListType tileSizes;
    tileSizes.push_back(spirvConfig->workgroupTileSizes);
    tileSizes.push_back(spirvConfig->subgroupTileSizes);
    tileSizes.push_back(spirvConfig->invocationTileSizes);
    if (!spirvConfig->convFilterTileSizes.empty()) {
      tileSizes.push_back(spirvConfig->convFilterTileSizes);
    }

    if (failed(setOpConfigAndEntryPointFnTranslation(
            funcOp, rootOperation, tileSizes,
            /*nativeVectorSizes=*/ArrayRef<int64_t>(), spirvConfig->pipeline,
            spirvConfig->workgroupSize))) {
      return failure();
    }

    // Propogate the `lowering.config` attribute to the other ops.
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

    if (spirvConfig->workgroupCount) {
      // Let the entry point region to return fully static number of workgroups.
      // This is needed for folding `affine.min` ops to expose static-shaped
      // tiled convolution for vectorization.
      // TODO(#5034): Use a proper way to prove tilability and fold
      // `affine.min`s.
      auto numWorkgroupsFn = [&](OpBuilder &b, Location loc,
                                 std::array<Value, 3>) {
        std::array<Value, 3> xyz;
        for (unsigned i = 0; i < 3; ++i) {
          auto count = spirvConfig->workgroupCount->at(i);
          xyz[i] = b.create<ConstantIndexOp>(loc, count);
        }
        return xyz;
      };

      OpBuilder builder(rootOperation->getContext());
      if (failed(defineWorkgroupCountRegion(builder, funcOp, numWorkgroupsFn)))
        return failure();
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
