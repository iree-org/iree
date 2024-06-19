// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"

namespace mlir::iree_compiler::IREE::Vulkan {

VulkanDialect::VulkanDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VulkanDialect>()) {
  registerAttributes();
}

} // namespace mlir::iree_compiler::IREE::Vulkan
