// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.cpp - Utility functions for SPIR-V CodeGen -------------------===//

#include "iree/compiler/Codegen/SPIRV/Utils.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

bool usesSPIRVCodeGen(IREE::HAL::ExecutableVariantOp variantOp) {
  if (variantOp.getObjects().has_value()) {
    // Variants containing external executables do not go through CodeGen.
    return false;
  }

  DictionaryAttr configuration = variantOp.getTargetAttr().getConfiguration();
  // The spirv.target_env attribute is attached if going down SPIR-V CodeGen
  // pipelines. Later we turn spirv.target_env into iree.spirv.features after
  // materializing device queries.
  return configuration.contains(spirv::getTargetEnvAttrName()) or
         configuration.contains("iree.spirv.features");
}

const char *getSPIRVDistributeAttrName() { return "iree.spirv.distribute_dim"; }

spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op) {
  auto variant = op->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variant)
    return nullptr;
  IREE::HAL::ExecutableTargetAttr targetAttr = variant.getTarget();
  if (!targetAttr)
    return nullptr;
  auto config = targetAttr.getConfiguration();
  if (!config)
    return nullptr;
  return config.getAs<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
}

std::optional<int> getSPIRVSubgroupSize(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  auto exportOp = exportOps.lookup(funcOp.getName());
  if (!exportOp)
    return std::nullopt;
  if (auto size = exportOp.getSubgroupSize())
    return size->getSExtValue();

  spirv::TargetEnvAttr target = getSPIRVTargetEnvAttr(funcOp);
  if (!target)
    return std::nullopt;
  return target.getResourceLimits().getSubgroupSize();
}

FailureOr<SmallVector<int64_t>> getSPIRVTileSize(func::FuncOp funcOp,
                                                 int tilingLevel) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  auto config = getLoweringConfig(computeOps);
  if (failed(config)) {
    return funcOp.emitOpError("failed to get lowering configuration");
  }

  return config->getTileSizeVals(tilingLevel);
}

FailureOr<linalg::TileSizeComputationFunction>
getSPIRVTileSizeComputeFn(func::FuncOp funcOp, int tilingLevel) {
  auto tileSizes = getSPIRVTileSize(funcOp, tilingLevel);
  if (failed(tileSizes))
    return failure();
  linalg::TileSizeComputationFunction computeFn =
      [tileSizes](OpBuilder &builder, Operation *op) {
        auto range = llvm::map_range(*tileSizes, [&](int64_t size) -> Value {
          return builder.create<arith::ConstantIndexOp>(op->getLoc(), size);
        });
        return llvm::to_vector(range);
      };
  return computeFn;
}

FailureOr<scf::SCFTileSizeComputationFunction>
getSPIRVScfTileSizeComputeFn(func::FuncOp funcOp, int tilingLevel) {
  FailureOr<SmallVector<int64_t>> tileSizes =
      getSPIRVTileSize(funcOp, tilingLevel);
  if (failed(tileSizes))
    return failure();
  scf::SCFTileSizeComputationFunction computeFn =
      [tileSizes](OpBuilder &builder,
                  Operation *op) -> SmallVector<OpFoldResult> {
    auto tileSizesOfr = getAsIndexOpFoldResult(op->getContext(), *tileSizes);
    auto zeroAttr = builder.getIndexAttr(0);
    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    tileSizesOfr.resize(numLoops, zeroAttr);
    return tileSizesOfr;
  };
  return computeFn;
}

template <typename GPUIdOp, typename GPUCountOp>
static linalg::ProcInfo
getGPUProcessorIdAndCountImpl(OpBuilder &builder, Location loc, unsigned dim) {
  assert(dim < kNumGPUDims && "processor index out of range!");

  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  Type indexType = builder.getIndexType();
  return {builder.create<GPUIdOp>(loc, indexType, dimAttr[dim]),
          builder.create<GPUCountOp>(loc, indexType, dimAttr[dim]),
          linalg::DistributionMethod::Cyclic};
}

template <typename GPUIdOp, typename GPUCountOp>
static SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCountsImpl(OpBuilder &builder, Location loc,
                                unsigned numDims) {
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] =
        getGPUProcessorIdAndCountImpl<GPUIdOp, GPUCountOp>(builder, loc, i);
  }
  return procInfo;
}

template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts(OpBuilder &builder,
                                                             Location loc,
                                                             unsigned numDims) {
  return getGPUProcessorIdsAndCountsImpl<GPUIdOp, GPUCountOp>(builder, loc,
                                                              numDims);
}

/// Explicit instantiation of gpuGPUProcessorIdsAndCounts.
template SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
    OpBuilder &builder, Location loc, unsigned numDims);

} // namespace mlir::iree_compiler
