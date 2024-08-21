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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

// Returns true if the given variant op uses SPIR-V CodeGen.
bool usesSPIRVCodeGen(IREE::HAL::ExecutableVariantOp variantOp);

/// Returns the attribute name carrying information about distribution.
const char *getSPIRVDistributeAttrName();

/// Given an operation, returns the HAL target config attribute.
DictionaryAttr getTargetConfigAttr(Operation *op);

/// Returns whether indirect bindings are supported based on the target config
/// applicable to the given |op|.
bool usesIndirectBindingsAttr(Operation *op);

/// Returns the tile sizes at the given `tilingLevel` for compute ops in
/// `funcOp`.
FailureOr<SmallVector<int64_t>>
getSPIRVTileSize(mlir::FunctionOpInterface funcOp, int tilingLevel);

/// Returns the functor to compute tile sizes at the given `tilingLevel` for
/// compute ops in `funcOp`.
FailureOr<linalg::TileSizeComputationFunction>
getSPIRVTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel);

/// Returns the functor to compute tile sizes at the given `tilingLevel` for
/// compute ops in `funcOp`.
FailureOr<scf::SCFTileSizeComputationFunction>
getSPIRVScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel);

/// Generate the operations that compute the processor ID and number of
/// processors. Used as the callback needed for LinalgDistributionOptions.
template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts(OpBuilder &builder, Location loc, unsigned numDims);

} // namespace mlir::iree_compiler

#endif //  IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
