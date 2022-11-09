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

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace iree_compiler {

/// Given an operation, return the `spirv.target_env` attribute.
spirv::TargetEnvAttr getSPIRVTargetEnvAttr(Operation *op);

/// Returns the attribute name carrying information about distribution.
const char *getSPIRVDistributeAttrName();

/// Propagates lowering configuration to all compute ops.
///
/// This is useful for passes that happen after bufferization and still use
/// single op Linalg tiling patterns. Under such circumstances, this is needed
/// to cover `linalg.fill` ops created during converting to destination style
/// for bufferization.
LogicalResult propagateLoweringConfigToComputeOps(func::FuncOp funcOp);

/// Generate the operations that compute the processor ID and number of
/// processors. Used as the callback needed for LinalgDistributionOptions.
template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts(OpBuilder &builder,
                                                             Location loc,
                                                             unsigned numDims);

}  // namespace iree_compiler
}  // namespace mlir

#endif  //  IREE_COMPILER_CODEGEN_SPIRV_UTILS_H_
