// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VULKAN_IR_VULKANTYPES_H_
#define IREE_COMPILER_DIALECT_VULKAN_IR_VULKANTYPES_H_

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Vulkan/IR/VulkanEnums.h.inc"  // IWYU pragma: export
// clang-format on

#endif  // IREE_COMPILER_DIALECT_VULKAN_IR_VULKANTYPES_H_
