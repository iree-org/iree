// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.cpp - Utility functions used in Linalg to SPIR-V lowering ----===//
//
// Implementaiton of utility functions used while lowering from Linalg to SPIRV.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/Utils.h"

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

const char *getSPIRVDistributeAttrName() { return "iree.spirv.distribute_dim"; }

spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op) {
  auto variant = op->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variant) return nullptr;
  IREE::HAL::ExecutableTargetAttr targetAttr = variant.getTarget();
  if (!targetAttr) return nullptr;
  auto config = targetAttr.getConfiguration();
  if (!config) return nullptr;
  return config.getAs<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
}

/// Returns true if the given MemRef is in workgroup memory.
bool isInWorkgroupMemory(MemRefType memrefType) {
  auto attribute =
      memrefType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
  if (attribute &&
      attribute.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace())
    return true;
  return false;
}

std::optional<int> getSPIRVSubgroupSize(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  auto exportOp = exportOps.lookup(funcOp.getName());
  if (!exportOp) return std::nullopt;
  if (auto size = exportOp.getSubgroupSize()) return size->getSExtValue();

  spirv::TargetEnvAttr target = getSPIRVTargetEnvAttr(funcOp);
  if (!target) return std::nullopt;
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

FailureOr<linalg::TileSizeComputationFunction> getSPIRVTileSizeComputeFn(
    func::FuncOp funcOp, int tilingLevel) {
  auto tileSizes = getSPIRVTileSize(funcOp, tilingLevel);
  if (failed(tileSizes)) return failure();
  linalg::TileSizeComputationFunction computeFn =
      [tileSizes](OpBuilder &builder, Operation *op) {
        auto range = llvm::map_range(*tileSizes, [&](int64_t size) -> Value {
          return builder.create<arith::ConstantIndexOp>(op->getLoc(), size);
        });
        return llvm::to_vector<4>(range);
      };
  return computeFn;
}

template <typename GPUIdOp, typename GPUCountOp>
static linalg::ProcInfo getGPUProcessorIdAndCountImpl(OpBuilder &builder,
                                                      Location loc,
                                                      unsigned dim) {
  assert(dim < kNumGPUDims && "processor index out of range!");

  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  Type indexType = builder.getIndexType();
  return {builder.create<GPUIdOp>(loc, indexType, dimAttr[dim]),
          builder.create<GPUCountOp>(loc, indexType, dimAttr[dim]),
          linalg::DistributionMethod::Cyclic};
}

template <typename GPUIdOp, typename GPUCountOp>
static SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCountsImpl(
    OpBuilder &builder, Location loc, unsigned numDims) {
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

}  // namespace iree_compiler
}  // namespace mlir
