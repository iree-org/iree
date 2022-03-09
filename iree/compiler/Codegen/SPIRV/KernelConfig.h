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

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

namespace detail {

/// Sets CodeGen configurations via attributes to the given convolution
/// `linalgOp` by trying to achieve the given `bestTilingFactor`, which is how
/// many scalar elements each thread should handle.
LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor);

/// Sets CodeGen configurations via attributes to the given matmul `linalgOp`
/// with the given best workgroup size and tile size hints.
LogicalResult setMatmulOpConfig(linalg::LinalgOp linalgOp,
                                std::array<int64_t, 2> bestWorkgroupSizeXY,
                                std::array<int64_t, 3> bestThreadTileSizeMNK);

/// Sets CodeGen configuration for GPUs from a specific vendor.
///
/// If the given `rootOp` has known good CodeGen configuration, attaches a
/// `translation_info` attribute to the entry point containing `rootOp` and a
/// `lowering_config` attribute to `rootOp`.
///
/// Returns success when either no configuration is found or a configuration is
/// successfullly attached as attribute. Returns failure only when there is an
/// issue attaching the attribute.

LogicalResult setAdrenoCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp);
LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                   Operation *rootOp);
LogicalResult setNVIDIACodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp);

}  // namespace detail

/// Attaches the `translation_info` attribute to entry points in `moduleOp` and
/// `lowering_config` attributes to all root ops in `moduleOp`'s region.
/// These attributes are used to drive the CodeGen pipeline.
LogicalResult initSPIRVLaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
