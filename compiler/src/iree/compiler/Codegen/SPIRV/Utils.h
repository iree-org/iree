// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Utils.h - Utility functions for lowering Linalg to SPIR-V ----------===//
//
// Utility functions used while lowering from Linalg to SPIR-V.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
#define IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

/// Returns the attribute name carrying information about distribution.
const char *getSPIRVDistributeAttrName();

/// Given an operation, returns the `spirv.target_env` attribute.
spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op);

/// Given a FuncOp, returns the subgroup size to use for CodeGen, by first
/// querying the hal.executable.export op, and then the SPIR-V target
/// environment. Returns std::nullopt on failures.
std::optional<int> getSPIRVSubgroupSize(func::FuncOp funcOp);

/// Returns the tile sizes at the given `tilingLevel` for compute ops in
/// `funcOp`.
FailureOr<SmallVector<int64_t>> getSPIRVTileSize(func::FuncOp funcOp,
                                                 int tilingLevel);

/// Returns the functor to compute tile sizes at the given `tilingLevel` for
/// compute ops in `funcOp`.
FailureOr<linalg::TileSizeComputationFunction>
getSPIRVTileSizeComputeFn(func::FuncOp funcOp, int tilingLevel);

/// Generate the operations that compute the processor ID and number of
/// processors. Used as the callback needed for LinalgDistributionOptions.
template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts(OpBuilder &builder, Location loc, unsigned numDims);

} // namespace mlir::iree_compiler

#endif //  IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
