// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_LLVMCPUSELECTUKERNELS_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_LLVMCPUSELECTUKERNELS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

// Returns a `UKernelDescriptorAttr` if a built-in LLVMCPU C ukernel applies
// to `op` AND `--iree-llvmcpu-enable-llvm-ukernels` enables the matching
// category, otherwise returns null. The caller is expected to attach the
// returned descriptor (e.g. via `setUKernelDescriptor`); attaching the
// bitcode itself is deferred to the `#iree_cpu.ukernel_provider`'s
// `createAndReplaceWithUkernelOp` at `LowerBitcodeUKernelsPass` time.
//
// The CPU side mirrors `compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/
// LLVMGPUSelectUKernels.cpp` but is simpler: there's no per-target-arch
// bitcode filename (the build features go in the filename via the
// embedded-TOC entry name) and no shared-memory sizing.
IREE::Codegen::UKernelDescriptorAttr selectCPUUKernel(Operation *op);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_LLVMCPUSELECTUKERNELS_H_
