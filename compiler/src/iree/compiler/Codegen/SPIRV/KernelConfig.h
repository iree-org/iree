// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- KernelConfig.h - Kernel Generation Configurations ------------------===//
//
// This file declares utility functions for configuring SPIR-V kernel
// generation, e.g., tiling schemes and workgroup size for important
// Linalg named ops.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
#define IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_

#include <array>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

/// By default don't do any pipelining.
constexpr unsigned defaultSimtSoftwarePipelineDepth = 1;
constexpr unsigned defaultSimtSoftwarePipelineStoreStage = 1;

constexpr unsigned defaultCoopMatrixSoftwarePipelineDepth = 1;
constexpr unsigned defaultCoopMatrixSoftwarePipelineStoreStage = 0;

/// Computes the total number of bytes if promoting both matmul LHS and RHS with
/// the tiven tile sizes.
int64_t getTileBytes(int64_t mTileSize, int64_t nTileSize, int64_t kTileSize,
                     int64_t elementBits, bool promoteC);

/// Adjusts the shared memory usage based on the pipelining depth.
int64_t getMultiBufferMemoryUsage(int64_t usedBytes, unsigned depth,
                                  unsigned storeStage);

/// Returns true if the given generic op is an elementwise op that can be fused
/// together with cooperative matrix in the same dispatch.
bool isCooperativeMatrixFusable(linalg::GenericOp genericOp);

/// Returns true if we need to promote C matrix to use cooperative matrix for
/// the the give matmul.
///
/// This is conservative by default--all unsupported cases will return true.
bool needToPrmoteCForCooperativeMatrix(linalg::LinalgOp matmulOp);

namespace detail {

const int bankConflictReductionPaddingBits = 128;

/// Sets CodeGen configurations via attributes to the given convolution
/// `linalgOp` by trying to achieve the given `bestTilingFactor`, which is how
/// many scalar elements each thread should handle.
LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor);

/// Sets CodeGen configurations via attributes to the given matmul `linalgOp`
/// with the given best workgroup size and tile size hints.
LogicalResult setMatmulOpConfig(
    spirv::ResourceLimitsAttr limits, linalg::LinalgOp linalgOp,
    std::array<int64_t, 2> bestWorkgroupSizeXY,
    std::array<int64_t, 3> bestThreadTileSizeMNK, bool enablePromotion = false,
    unsigned softwarePipelineDepth = defaultSimtSoftwarePipelineDepth,
    unsigned softwarePipelineStoreStage =
        defaultSimtSoftwarePipelineStoreStage);

/// Sets CodeGen configurations via attributes to the given matmul `linalgOp`
/// with tile sizes for cooperative matrix, if possible for the given matmul
/// size.
LogicalResult setCooperativeMatrixConfig(
    const spirv::TargetEnv &targetEnv, linalg::LinalgOp op,
    const unsigned numSubgroupsPerWorkgroup,
    const unsigned numMNTilesPerSubgroup,
    unsigned softwarePipelineDepth = defaultCoopMatrixSoftwarePipelineDepth,
    unsigned softwarePipelineStoreStage =
        defaultCoopMatrixSoftwarePipelineStoreStage);

/// Sets CodeGen configuration for GPUs from a specific vendor.
///
/// If the given `rootOp` has known good CodeGen configuration, attaches a
/// `translation_info` attribute to the entry point containing `rootOp` and a
/// `lowering_config` attribute to `rootOp`.
///
/// Returns success when a configuration is successfullly attached as attribute.
/// Returns failure otherwise.

LogicalResult setAdrenoCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp);
LogicalResult setAppleCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                    Operation *rootOp);
LogicalResult setAMDCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                  Operation *rootOp);
LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                   Operation *rootOp);
LogicalResult setNVIDIACodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp);

}  // namespace detail

/// Returns true if the given `linalgOp` is a (batch) matmul op.
bool isMatmulOrBatchMatmul(linalg::LinalgOp linalgOp);

/// Given the linalg `op` with `lhsShape` and `rhsShape`, tries to treat as a
/// (batch) matmul like op and deduce the index of the loop corresponding to
/// B/M/N/K dimension respectively. Returns -1 as the index if unable to deduce.
std::tuple<int, int, int, int> getMatmulBMNKIndex(
    linalg::LinalgOp op, int *lastParallelDim = nullptr);

/// Attaches the `translation_info` attribute to entry points in `moduleOp` and
/// `lowering_config` attributes to all root ops in `moduleOp`'s region.
/// These attributes are used to drive the CodeGen pipeline.
LogicalResult initSPIRVLaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
