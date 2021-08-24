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

struct SPIRVCodeGenConfig {
  /// The pipeline to use for SPIR-V CodeGen.
  IREE::HAL::DispatchLoweringPassPipeline pipeline;

  /// Tiling sizes along each level. The following are all optional; if empty,
  /// it means no tiling along the particular level.
  SmallVector<int64_t, 4> workgroupTileSizes;
  SmallVector<int64_t, 4> subgroupTileSizes;
  SmallVector<int64_t, 4> invocationTileSizes;
  SmallVector<int64_t, 4> convFilterTileSizes;

  /// Workgroup size and count. Workgroup count is optional. If existing, it
  /// will override the default workgorup count settings on entry point ops.
  std::array<int64_t, 3> workgroupSize;
  // TODO(#5034): The workgroup count shouldn't exist. It's now needed for
  // exposing static shape for convolution CodeGen. Remove this after we have
  // a better way there.
  llvm::Optional<std::array<int64_t, 3>> workgroupCount;
};

/// Returns a SPIR-V CodeGen configuration for the given target environment and
/// operation. Returns llvm::None if there is no configuration.
llvm::Optional<SPIRVCodeGenConfig> getMaliCodeGenConfig(
    const spirv::TargetEnv &targetEnv, Operation *op);
llvm::Optional<SPIRVCodeGenConfig> getNVIDIACodeGenConfig(
    const spirv::TargetEnv &targetEnv, Operation *op);

}  // namespace detail

/// Attaches the `translation.info` attribute to entry points in `moduleOp` and
/// `lowering.config` attributes to all root ops in `moduleOp`'s region.
/// These attributes are used to drive the CodeGen pipeline.
LogicalResult initSPIRVLaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_SPIRV_KERNELCONFIG_H_
