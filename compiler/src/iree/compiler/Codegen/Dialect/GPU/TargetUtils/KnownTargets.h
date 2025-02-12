// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_KNOWNTARGETS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_KNOWNTARGETS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::IREE::GPU {

// Returns a TargetAttr to target Metal via SPIR-V CodeGen.
TargetAttr getMetalTargetDetails(MLIRContext *context);

// Returns a TargetAttr to describe the details of the given |target|, which can
// be a product name like "rtx3090", an microarchitecture name like "ampere", or
// a compute capability like "sm_80", with a list of comma-separated target
// |features|. Returns a null TargetAttr if the given |target| is not
// recognized.
TargetAttr getCUDATargetDetails(llvm::StringRef target,
                                llvm::StringRef features, MLIRContext *context);

// Normalizes the given CUDA |target| to the gfx target commonly used for
// compiling towards CUDA. For example, "sm_80" for "a100", "sm_89" for "ada".
// if the given |target| is not recognized.
StringRef normalizeCUDATarget(StringRef target);

// Returns a TargetAttr to describe the details of the given |target|, which can
// be a product name like "rx7900xtx", an microarchitecture name like "rdna3",
// or a compiler target like "gfx1100", with a list of comma-separated
// target |features|. Returns a null TargetAttr if the given |target| is not
// recognized.
TargetAttr getHIPTargetDetails(llvm::StringRef target, llvm::StringRef features,
                               MLIRContext *context);

// Returns an attribute implementing `EncodingLayoutAttributeInterface` if
// |target| has known encoding preferences.
Attribute getHIPTargetEncodingLayoutAttr(TargetAttr target);

// Normalizes the given HIP |target| to the gfx target commonly used for
// compiling towards HIP. For example, "gfx90a" for "cnda2", "gfx1100" for
// "rx7900xtx". Returns empty StringRef if the given |target| is not recognized.
StringRef normalizeHIPTarget(StringRef target);

// Returns a TargetAttr to describe the details of the given |target|, which can
// be a product name like "rtx3090"/"mali-g710"/"adreno" or an microarchitecture
// name like "ampere"/"valhall". Returns a null TargetAttr if the given |target|
// is not recognized.
TargetAttr getVulkanTargetDetails(llvm::StringRef target, MLIRContext *context);

// Returns a TargetAttr to target WebGPU via SPIR-V CodeGen.
TargetAttr getWebGPUTargetDetails(MLIRContext *context);

// Returns the full target of the given |aliasTarget| with a list of
// comma-separated target |features|. Returns null target if unknown.
TargetAttr getFullTarget(StringRef targetAPI, StringRef aliasTarget,
                         llvm::StringRef features, MLIRContext *context);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_KNOWNTARGETS_H_
