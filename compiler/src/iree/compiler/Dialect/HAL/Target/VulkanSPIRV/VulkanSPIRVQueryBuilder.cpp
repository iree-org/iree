// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVQueryBuilder.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Return the subgroup feature bit according to the following table:
//
// VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001
// VK_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002
// VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004
// VK_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008
// VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010
// VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x00000020
// VK_SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040
// VK_SUBGROUP_FEATURE_QUAD_BIT = 0x00000080
// VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV = 0x00000100
uint32_t processSubgroupCapability(spirv::Capability c) {
  switch (c) {
  case spirv::Capability::GroupNonUniform:
    return 0x00000001;
  case spirv::Capability::GroupNonUniformVote:
    return 0x00000002;
  case spirv::Capability::GroupNonUniformArithmetic:
    return 0x00000004;
  case spirv::Capability::GroupNonUniformBallot:
    return 0x00000008;
  case spirv::Capability::GroupNonUniformShuffle:
    return 0x00000010;
  case spirv::Capability::GroupNonUniformShuffleRelative:
    return 0x00000020;
  case spirv::Capability::GroupNonUniformClustered:
    return 0x00000040;
  case spirv::Capability::GroupNonUniformQuad:
    return 0x00000080;
  case spirv::Capability::GroupNonUniformPartitionedNV:
    return 0x00000100;
  default:
    break;
  }
  return 0x00000000;
}

LogicalResult buildVulkanSPIRVDeviceRequirementQueries(
    IREE::HAL::ExecutableVariantOp variantOp) {

  MLIRContext *context = variantOp.getContext();
  DictionaryAttr configuration = variantOp.getTarget().getConfiguration();
  spirv::TargetEnvAttr defaultEnvAttr = convertTargetEnv(
      Vulkan::getTargetEnvForTriple(context, "unknown-unknown-unknown"));
  spirv::TargetEnv defaultEnv(defaultEnvAttr);

  uint32_t subgroupCapabilities = 0x00000001;
  auto hasModule = false;
  variantOp.walk([&](spirv::ModuleOp spirvModuleOp) {
    hasModule = true;
    if (spirvModuleOp.getVceTriple()) {
      for (spirv::Capability capability :
           spirvModuleOp.getVceTriple()->getCapabilities()) {
        // No need to check if already allowed by the default environment.
        if (defaultEnv.allows(capability)) {
          continue;
        }
        subgroupCapabilities |= processSubgroupCapability(capability);
      }
    }
  });

  // If there is no module present, assume this is an external dispatch. In this
  // case we pessimistically use the capabilities/extensions specified on the
  // variant target attribute.
  if (!hasModule) {
    StringAttr spvEnvName =
        Builder(context).getStringAttr(spirv::getTargetEnvAttrName());
    Attribute maybeSpvEnv = configuration.get(spvEnvName);
    // If no attribute present, assume a default configuration.
    if (!maybeSpvEnv) {
      return success();
    }
    spirv::TargetEnvAttr variantTargetEnv =
        dyn_cast<spirv::TargetEnvAttr>(maybeSpvEnv);
    // There is a target env, but it isn't a SPIR-V one ...
    if (!variantTargetEnv) {
      return failure();
    }

    for (auto capability : variantTargetEnv.getCapabilities()) {
      if (defaultEnv.allows(capability)) {
        continue;
      }
      subgroupCapabilities |= processSubgroupCapability(capability);
    }
  }

  // Check if nothing is needed beyond the basic bit.
  if (subgroupCapabilities == 0x00000001) {
    return success();
  }

  // Begin constructing the conditional region. We wait until the end to avoid
  // creating unless needed.
  OpBuilder builder(variantOp);
  Value device = variantOp.emplaceConditionOp(builder);
  Location loc = device.getLoc();

  auto i1Type = builder.getI1Type();
  auto subgroupQuery = builder.create<IREE::HAL::DeviceQueryOp>(
      loc, i1Type, i1Type, device,
      builder.getStringAttr("hal.device.vulkan.subgroup_operations"),
      builder.getStringAttr(std::to_string(subgroupCapabilities)),
      builder.getZeroAttr(i1Type));
  // Verify that the query succeeds and indicates the correct support.
  auto querySuccess = builder.create<arith::AndIOp>(
      loc, subgroupQuery.getResult(0), subgroupQuery.getResult(1));
  builder.create<IREE::HAL::ReturnOp>(loc, querySuccess.getResult());
  return success();
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
