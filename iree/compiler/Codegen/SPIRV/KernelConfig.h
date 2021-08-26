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

/// If the given `op` has known good CodeGen configuration for Mali GPUs,
/// attaches a `translation.info` attribute to the entry point containing `op`
/// and a `lowering.config` attribute to `op`. Returns the status of setting the
/// attributes. Does nothing and returns llvm::None otherwise.
Optional<LogicalResult> setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                             Operation *op);

/// If the given `op` has known good CodeGen configuration for NVIDIA GPUs,
/// attaches a `translation.info` attribute to the entry point containing `op`
/// and a `lowering.config` attribute to `op`. Returns the status of setting the
/// attributes. Does nothing and returns llvm::None otherwise.
Optional<LogicalResult> setNVIDIACodeGenConfig(
    const spirv::TargetEnv &targetEnv, Operation *op);

}  // namespace detail

/// Attaches the `translation.info` attribute to entry points in `moduleOp` and
/// `lowering.config` attributes to all root ops in `moduleOp`'s region.
/// These attributes are used to drive the CodeGen pipeline.
LogicalResult initSPIRVLaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
