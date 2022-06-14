// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_WEBGPUTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_WEBGPUTARGET_H_

#include <functional>
#include <string>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Options controlling the WebGPU/WGSL translation.
struct WebGPUTargetOptions {
  // Include debug information like variable names in outputs.
  bool debugSymbols = true;
};

// Returns a WebGPUTargetOptions struct initialized with WebGPU/WGSL related
// command-line flags.
WebGPUTargetOptions getWebGPUTargetOptionsFromFlags();

// Registers the WebGPU/WGSL backends.
void registerWebGPUTargetBackends(
    std::function<WebGPUTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_WEBGPUTARGET_H_
