// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_
#define IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_

#include <string>

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::Vulkan {

/// GPU triple definitions to describe GPU targets for compilers.
///
/// We use "triple" here to match common compiler language: historically one
/// would describe a CPU compiler target as a string containing exactly three
/// fields. But here the configuration is for GPU and there can exist a lot of
/// architectures/vendors/products/systems. What matters differ from CPU
/// triples. We define it in the form of:
///
///   <vendor/arch>-<product>-<os>
///
/// For example:
///   ampere-rtx3080-windows
///   rdna1-5700xt-linux
///   adreno-a650-android30
///   valhall-unknown-android30
///   cpu-swiftshader-unknown
///
/// Vendor and architecture are combined together because:
/// * Typically each GPU vendor has its own set of architectures. So given the
///   architecture we know which vendor it is from. This is different from CPU
///   land where the the same architecture can be implemented by mulitple
///   vendors.
/// * There are vendors that we don't have public information regarding its
///   architectures.
/// We need a field for product to differentiate the cases where the
/// architecture is unknown or ambiguous.
class TargetTriple {
public:
  static TargetTriple get(const char *triple);

  TargetTriple(TargetTripleArch, TargetTripleProduct, TargetTripleOS);

  TargetTripleArch getArch() const { return arch; }
  TargetTripleProduct getProduct() const { return product; }
  TargetTripleOS getOS() const { return os; }

  /// Returns the triple string.
  std::string getTriple() const;

  TargetEnvAttr getTargetEnv(MLIRContext *context) const;

private:
  TargetTripleArch arch;
  TargetTripleProduct product;
  TargetTripleOS os;
};

} // namespace mlir::iree_compiler::IREE::Vulkan

#endif // IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_
