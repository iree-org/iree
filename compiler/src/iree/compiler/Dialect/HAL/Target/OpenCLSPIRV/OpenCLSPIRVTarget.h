// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_OPENCLSPIRV_OPENCLSPIRVTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_OPENCLSPIRV_OPENCLSPIRVTARGET_H_

#include <functional>
#include <string>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Options controlling the SPIR-V translation.
struct OpenCLSPIRVTargetOptions {
  // OpenCL target triple.
  std::string openCLTargetTriple;
};

// Returns a OpenCLSPIRVTargetOptions struct initialized with OpenCL/SPIR-V
// related command-line flags.
OpenCLSPIRVTargetOptions getOpenCLSPIRVTargetOptionsFromFlags();

// Registers the OpenCL/SPIR-V backends.
void registerOpenCLSPIRVTargetBackends(
    std::function<OpenCLSPIRVTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_OPENCLSPIRV_OPENCLSPIRVTARGET_H_
