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

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

namespace detail {

/// Lets the entry point region to return fully static number of workgroups.
// This is needed for folding `affine.min` ops to expose static-shaped tiled
// convolution for vectorization.
// TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
LogicalResult defineConvWorkgroupCountRegion(
    Operation *op, ArrayRef<int64_t> outputShape,
    ArrayRef<int64_t> workgroupTileSizes);

/// Sets CodeGen configuration for GPUs from a specific vendor.
///
/// If the given `rootOp` has known good CodeGen configuration, attaches a
/// `translation.info` attribute to the entry point containing `rootOp` and a
/// `lowering.config` attribute to `rootOp`.
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

/// Attaches the `translation.info` attribute to entry points in `moduleOp` and
/// `lowering.config` attributes to all root ops in `moduleOp`'s region.
/// These attributes are used to drive the CodeGen pipeline.
LogicalResult initSPIRVLaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
