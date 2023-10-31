// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVQUERYBUILDER_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVQUERYBUILDER_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Helper to build the required set of device queries to check whether the
// device supports the given configuration.
LogicalResult buildVulkanSPIRVDeviceRequirementQueries(
    IREE::HAL::ExecutableVariantOp variantOp);

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVQUERYBUILDER_H_
