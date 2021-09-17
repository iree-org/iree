// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Given `nprocs`, tries to distribute it evenly across 2 logical dimensions.
static std::tuple<int64_t, int64_t> distributeProcs2D(int64_t nprocs) {
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
// Convolution Default Configuration
//===----------------------------------------------------------------------===//

/// Lets the entry point region to return fully static number of workgroups.
// This is needed for folding `affine.min` ops to expose static-shaped tiled
// convolution for vectorization.
// TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
static LogicalResult defineConvWorkgroupCountRegion(
    Operation *op, ArrayRef<int64_t> outputShape,
    ArrayRef<int64_t> workgroupTileSizes) {
  auto numWorkgroupsFn = [&](OpBuilder &b, Location loc, std::array<Value, 3>) {
    std::array<Value, 3> xyz;
    for (unsigned i = 0; i < 3; ++i) {
      int64_t count = outputShape[i] / workgroupTileSizes[i];
      // This is meant for perfectly tilable cases. Double check that.
      assert(outputShape[i] % workgroupTileSizes[i] == 0 && count != 0);
      xyz[2 - i] = b.create<ConstantIndexOp>(loc, count);
    }
    return xyz;
  };
  OpBuilder builder(op->getContext());
  return defineWorkgroupCountRegion(builder, op->getParentOfType<FuncOp>(),
                                    numWorkgroupsFn);
}

namespace detail {

LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor) {
  ArrayRef<int64_t> inputShape = getUntiledShape(linalgOp.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);
  if (llvm::any_of(inputShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(outputShape, ShapedType::isDynamic)) return success();

  int64_t ic = inputShape[3];
  int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // The conversion pipeline requires the input channel dimension to be some
  // multipler of four, or less than four.
  if (!(ic % 4 == 0 || ic < 4)) return success();

  // The core idea is to distribute the convolution OH/OW/OC dimension to the
  // workgroup Z/Y/X dimension, with each thread in a workgroup handling
  // multiple vector elements. We try to 1) utilize all threads in a subgroup,
  // and 2) handle an optimal tile size along each dimension.

  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1);        // (X, Y, Z)
  SmallVector<int64_t, 4> workgroupTileSizes(4, 0);   // (N, OH, OW, OC)
  SmallVector<int64_t, 4> invocationTileSizes(4, 0);  // (N, OH, OW, OC)

  // Deduce the configuration for the OC dimension.
  for (int64_t x = residualThreads; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = 4;
    if (oc % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[3] = x * chosenTileSize;
      invocationTileSizes[3] = chosenTileSize;
      residualThreads /= x;
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[3] == 0) return success();

  // Deduce the configruation for the OW and OH dimension. Try to make them even
  // if possible given we typically have images with the same height and width.
  bool tileToSquare = false;
  unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (ow == oh && residualThreads != 1 && log2Threads % 2 == 0) {
    int64_t yz = 1 << (log2Threads / 2);

    int64_t chosenTileSize = 1 << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      workgroupSize[1] = workgroupSize[2] = yz;
      workgroupTileSizes[2] = workgroupTileSizes[1] = yz * chosenTileSize;
      invocationTileSizes[2] = invocationTileSizes[1] = chosenTileSize;
      tileToSquare = true;
    }
  }

  // Otherwise treat OW and OH separately to allow them to have different number
  // of threads and tiling size.
  if (!tileToSquare) {
    // Decide the tiling and distribution parameters for one dimension.
    auto decideOneDim = [&](int64_t inputDim, int64_t &wgDimSize,
                            int64_t &wgTileSize, int64_t &invoTileSize) {
      for (int64_t dim = residualThreads; dim >= 1; dim >>= 1) {
        int64_t chosenTileSize = 0;
        for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
          if (inputDim % (dim * t) == 0) {
            chosenTileSize = t;
            break;
          }
        }
        if (chosenTileSize) {
          wgDimSize = dim;
          wgTileSize = dim * chosenTileSize;
          invoTileSize = chosenTileSize;
          residualThreads /= dim;
          residualTilingFactor /= chosenTileSize;
          return true;
        }
      }
      return false;
    };

    if (!decideOneDim(ow, workgroupSize[1], workgroupTileSizes[2],
                      invocationTileSizes[2]) ||
        !decideOneDim(oh, workgroupSize[2], workgroupTileSizes[1],
                      invocationTileSizes[1])) {
      return success();
    }
  }

  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.emplace_back();  // Subgroup level
  tileSizes.push_back(invocationTileSizes);

  auto funcOp = linalgOp->getParentOfType<FuncOp>();
  if (failed(setOpConfigAndEntryPointFnTranslation(
          funcOp, linalgOp, tileSizes, {}, pipeline, workgroupSize))) {
    return failure();
  }
  return defineConvWorkgroupCountRegion(
      linalgOp, llvm::makeArrayRef(outputShape).drop_front(),
      llvm::makeArrayRef(workgroupTileSizes).drop_front());
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setOpConfig(spirv::ResourceLimitsAttr limits,
                                 linalg::BatchMatmulOp op) {
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

  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  TileSizesListType tileSizes;
  // Workgroup level.
  tileSizes.push_back({numBatchesPerThread, numRowsPerThread * workgroupSize[1],
                       numColsPerThread * workgroupSize[0], tileSizeK});
  // No tiling at the subgroup level since this target doesn't use subgroup op
  // or shared memory.
  tileSizes.emplace_back();
  // Invocation level.
  tileSizes.push_back(
      {numBatchesPerThread, numRowsPerThread, numColsPerThread, 0});

  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

static LogicalResult setOpConfig(spirv::ResourceLimitsAttr limits,
                                 linalg::MatmulOp op) {
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

  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  TileSizesListType tileSizes;
  // Workgroup level.
  tileSizes.push_back(
      {getMinIfStaticShape(M, numRowsPerThread * workgroupSize[1]),
       getMinIfStaticShape(N, numColsPerThread * workgroupSize[0]),
       getMinIfStaticShape(K, tileSizeK)});
  // No tiling at the subgroup level since this target doesn't use subgroup op
  // or shared memory.
  tileSizes.emplace_back();
  // Invocation level.
  tileSizes.push_back({1, 1, 0});

  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

static LogicalResult setOpConfig(spirv::ResourceLimitsAttr limits,
                                 linalg_ext::FftOp op) {
  const int64_t subgroupSize = limits.subgroup_size().getValue().getSExtValue();
  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute;

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  auto partitionedLoops = getPartitionedLoops(op);
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t, 4> workgroupTileSize(loopDepth, 0);

  // Tiling along partitioned loops with size 1.
  for (int64_t loopIndex : partitionedLoops) {
    workgroupTileSize[loopIndex] = 1;
  }
  auto rank = op.getOperandRank();
  if (workgroupTileSize.size() >= rank && workgroupTileSize[rank - 1] != 0) {
    APInt value;
    if (matchPattern(op.getStage(), m_ConstantInt(&value))) {
      workgroupTileSize[rank - 1] = 1 << value.getSExtValue();
    } else {
      op.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSize};
  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setDefaultOpConfig(spirv::ResourceLimitsAttr limits,
                                        Operation *op) {
  auto partitionedLoops = getPartitionedLoops(op);
  if (partitionedLoops.empty()) {
    auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    auto funcOp = op->getParentOfType<FuncOp>();
    return setOpConfigAndEntryPointFnTranslation(funcOp, op, {}, {}, pipeline,
                                                 workgroupSize);
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

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSize);
  tileSizes.emplace_back();  // Subgroup level.
  tileSizes.push_back(threadTileSize);

  return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                               op, tileSizes, {}, pipeline,
                                               workgroupSize);
}

/// Sets the CodeGen configuration as attributes to the given `rootOp` if it's a
/// known Linalg matmul/convolution op with good configurations.
static LogicalResult setSPIRVOpConfig(const spirv::TargetEnv &targetEnv,
                                      Operation *rootOp) {
  LogicalResult result = success();
  // First try to find a proper CodeGen configuration to tile and vectorize for
  // the current target architecture.
  switch (targetEnv.getVendorID()) {
    case spirv::Vendor::ARM:
      result = detail::setMaliCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::NVIDIA:
      result = detail::setNVIDIACodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::Qualcomm:
      result = detail::setAdrenoCodeGenConfig(targetEnv, rootOp);
      break;
    default:
      break;
  }

  if (failed(result)) return result;
  // Check whether there is actually a configuration found. If so, it's done.
  if (getLoweringConfig(rootOp)) return result;

  // Otherwise fallback to use a default configuration.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp, linalg_ext::FftOp>(
          [limits](auto op) { return setOpConfig(limits, op); })
      .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwOp>(
          [limits](auto op) {
            // Try to tile and vectorize first. It's common to see 32 threads
            // per subgroup for GPUs.
            auto result = detail::setConvOpConfig(op, /*subgroupSize=*/32,
                                                  /*bestTilingFactor=*/32);
            if (failed(result)) return result;
            if (getLoweringConfig(op)) return result;

            // If unsuccessful, try to tile and distribute.
            return setDefaultOpConfig(limits, op);
          })
      .Default([](Operation *) { return success(); });
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult initSPIRVLaunchConfig(ModuleOp module) {
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps =
      getAllEntryPoints(module);
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(module);
  if (!targetEnvAttr) {
    return module.emitOpError(
        "expected parent hal.executable.variant to have spv.target_env "
        "attribute");
  }
  spirv::TargetEnv targetEnv(targetEnvAttr);
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();

  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPointOps.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    if (getTranslationInfo(entryPointOp)) continue;

    SmallVector<Operation *, 4> computeOps;
    SmallVector<Operation *, 4> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    int64_t subgroupSize =
        targetEnv.getResourceLimits().subgroup_size().getValue().getSExtValue();

    // If the dispatch region does not contain tiled and distributed Linalg ops,
    // invoke the pipeline to distribute to global invocations.
    if (tiledLoops.empty() && llvm::none_of(computeOps, [](Operation *op) {
          return hasMarker(op, getWorkgroupMarker());
        })) {
      std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
      if (failed(
              setTranslationUsingDistributeToGlobalId(funcOp, workgroupSize))) {
        return computeOps[0]->emitOpError(
            "failed to set translation info for distributing to global IDs");
      }
      continue;
    }

    Operation *rootOperation = nullptr;

    // Try to find a configuration according to a matmul/convolution op and use
    // it as the root op.
    for (Operation *computeOp : computeOps) {
      if (failed(setSPIRVOpConfig(targetEnv, computeOp))) return failure();

      // Check if the op configuration was set.
      if (!getLoweringConfig(computeOp)) continue;

      if (rootOperation) {
        return computeOp->emitOpError(
            "unhandled multiple roots in dispatch region");
      }
      rootOperation = computeOp;
    }

    // If there are still no root op, check for any linalg.generic op.
    if (!rootOperation) {
      for (Operation *computeOp : reverse(computeOps)) {
        if (failed(setDefaultOpConfig(limits, computeOp))) return failure();

        // Check if the op configuration was set.
        if (!getLoweringConfig(computeOp)) {
          return computeOp->emitOpError(
              "without known roots, the last operation in the tiled loop body "
              "is expected to be set as root");
        }
        rootOperation = computeOp;
        break;
      }
    }

    if (!rootOperation) {
      // If the tiled loops are not empty then this could be a corner case of
      // tensor.insert_slice being tiled and distributed, that just shows up as
      // a `flow.dispatch.tensor.load` and a `flow.dispatch.tensor.store` (or as
      // a copy. For now just treat the tiled loops not being empty as an
      // indicator of that. Need a better way of information flow from flow
      // dialect to hal.
      if (!tiledLoops.empty()) {
        const int64_t subgroupSize =
            limits.subgroup_size().getValue().getSExtValue();
        std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
        SmallVector<int64_t> workloadPerWorkgroup(tiledLoops.size(), 1);
        workloadPerWorkgroup.front() = subgroupSize * 4;
        setTranslationInfo(
            funcOp, IREE::HAL::DispatchLoweringPassPipeline::SPIRVDistribute,
            workgroupSize, workloadPerWorkgroup);
        return success();
      }
      return funcOp.emitError("contains no root Linalg operation");
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
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
