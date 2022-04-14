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

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

const char *getSPIRVDistributeAttrName() { return "iree.spirv.distribute_dim"; }

spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op) {
  auto variant = op->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variant) return nullptr;
  IREE::HAL::ExecutableTargetAttr targetAttr = variant.target();
  if (!targetAttr) return nullptr;
  auto config = targetAttr.getConfiguration();
  if (!config) return nullptr;
  return config.getAs<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
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
          builder.create<GPUCountOp>(loc, indexType, dimAttr[dim])};
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
