// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETENVIRONMENT_H_
#define IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETENVIRONMENT_H_

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

namespace mlir::iree_compiler::IREE::Vulkan {

/// Returns the Vulkan target environment attribute for the given GPU triple.
Vulkan::TargetEnvAttr getTargetEnvForTriple(MLIRContext *context,
                                            llvm::StringRef triple);

/// Converts the given Vulkan target environment into the corresponding SPIR-V
/// target environment.
///
/// Vulkan and SPIR-V are two different domains working closely. A Vulkan target
/// environment specifies the Vulkan version, extensions, features, and resource
/// limits queried from a Vulkan implementation. These properties typically have
/// corresponding SPIR-V bits, directly or indirectly. For example, by default,
/// Vulkan 1.0 supports SPIR-V 1.0 and Vulkan 1.1 supports up to SPIR-V 1.3.
/// If the VK_KHR_spirv_1_4 extension is available, then SPIR-V 1.4 can be used.
/// Similarly, if the VK_KHR_variable_pointers extension is available, then
/// the VariablePointersStorageBuffer capabilities on SPIR-V side can be
/// activated. The function handles the mapping relationship between tese two
/// domains.
spirv::TargetEnvAttr convertTargetEnv(Vulkan::TargetEnvAttr vkTargetEnv);

} // namespace mlir::iree_compiler::IREE::Vulkan

#endif // IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETENVIRONMENT_H_
