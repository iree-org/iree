// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/spirv.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

static constexpr uint32_t kComputeBdaModule[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // OpMemoryModel PhysicalStorageBuffer64 GLSL450
    0x0003000eu,
    5348u,
    1u,
    // OpEntryPoint GLCompute %1 "main"
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // OpExecutionMode %1 LocalSize 4 5 6
    0x00060010u,
    1u,
    17u,
    4u,
    5u,
    6u,
};

static constexpr uint32_t
    kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability[] = {
        0x07230203u,
        0x00010600u,
        0u,
        8u,
        0u,
        // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
        0x0003000eu,
        5348u,
        1u,
        // Declares OpEntryPoint GLCompute %1 "main".
        0x0005000fu,
        5u,
        1u,
        0x6e69616du,
        0u,
        // Declares OpExecutionMode %1 LocalSize 4 5 6.
        0x00060010u,
        1u,
        17u,
        4u,
        5u,
        6u,
};

TEST(SpirvTest, ParsesComputeEntryPoint) {
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_verify_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule)));

  bool uses_physical_storage_buffer64_glsl450 = false;
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_uses_physical_storage_buffer64_glsl450(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
      &uses_physical_storage_buffer64_glsl450));
  EXPECT_TRUE(uses_physical_storage_buffer64_glsl450);

  iree_host_size_t entry_point_count = 0;
  iree_host_size_t name_storage_size = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_count_compute_entry_points(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &entry_point_count,
      &name_storage_size));
  EXPECT_EQ(1u, entry_point_count);
  EXPECT_EQ(5u, name_storage_size);

  iree_hal_vulkan_spirv_compute_entry_point_t entry_point = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_parse_compute_entry_points(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
      /*entry_point_capacity=*/1, &entry_point));
  EXPECT_EQ(1u, entry_point.id);
  EXPECT_TRUE(iree_string_view_equal(entry_point.name, IREE_SV("main")));
  EXPECT_EQ(4u, entry_point.workgroup_size[0]);
  EXPECT_EQ(5u, entry_point.workgroup_size[1]);
  EXPECT_EQ(6u, entry_point.workgroup_size[2]);

  bool entry_point_found = false;
  uint32_t workgroup_size[3] = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_parse_compute_workgroup_size(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), IREE_SV("main"),
      &entry_point_found, workgroup_size));
  EXPECT_TRUE(entry_point_found);
  EXPECT_EQ(4u, workgroup_size[0]);
  EXPECT_EQ(5u, workgroup_size[1]);
  EXPECT_EQ(6u, workgroup_size[2]);
}

TEST(SpirvTest, RejectsTruncatedInstruction) {
  static constexpr uint32_t kTruncatedModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpMemoryModel declares four words but only has two trailing words.
      0x0004000eu,
      5348u,
  };
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_module(kTruncatedModule,
                                          IREE_ARRAYSIZE(kTruncatedModule)));
}

TEST(SpirvTest, ReportsMissingPhysicalStorageBufferMemoryModel) {
  static constexpr uint32_t kLogicalModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpMemoryModel Logical GLSL450
      0x0003000eu,
      0u,
      1u,
  };
  bool uses_physical_storage_buffer64_glsl450 = true;
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_uses_physical_storage_buffer64_glsl450(
      kLogicalModule, IREE_ARRAYSIZE(kLogicalModule),
      &uses_physical_storage_buffer64_glsl450));
  EXPECT_FALSE(uses_physical_storage_buffer64_glsl450);
}

TEST(SpirvTest, DetectsPhysicalStorageBufferAddressesCapability) {
  bool has_capability = false;
  IREE_ASSERT_OK(
      iree_hal_vulkan_spirv_has_physical_storage_buffer_addresses_capability(
          kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
          &has_capability));
  EXPECT_TRUE(has_capability);

  IREE_ASSERT_OK(
      iree_hal_vulkan_spirv_has_physical_storage_buffer_addresses_capability(
          kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability,
          IREE_ARRAYSIZE(
              kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability),
          &has_capability));
  EXPECT_FALSE(has_capability);
}

TEST(SpirvTest, DetectsDescriptorBindingDecorations) {
  static constexpr uint32_t kDescriptorDecoratedModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpDecorate %1 DescriptorSet 0
      0x00040047u,
      1u,
      34u,
      0u,
      // OpDecorate %1 Binding 2
      0x00040047u,
      1u,
      33u,
      2u,
  };
  bool has_descriptor_binding_decorations = false;
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_has_descriptor_binding_decorations(
      kDescriptorDecoratedModule, IREE_ARRAYSIZE(kDescriptorDecoratedModule),
      &has_descriptor_binding_decorations));
  EXPECT_TRUE(has_descriptor_binding_decorations);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_has_descriptor_binding_decorations(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
      &has_descriptor_binding_decorations));
  EXPECT_FALSE(has_descriptor_binding_decorations);
}

TEST(SpirvTest, RejectsDuplicateComputeEntryNames) {
  static constexpr uint32_t kDuplicateEntryModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpEntryPoint GLCompute %1 "main"
      0x0005000fu,
      5u,
      1u,
      0x6e69616du,
      0u,
      // OpEntryPoint GLCompute %2 "main"
      0x0005000fu,
      5u,
      2u,
      0x6e69616du,
      0u,
  };
  iree_hal_vulkan_spirv_compute_entry_point_t entry_points[2] = {};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_parse_compute_entry_points(
          kDuplicateEntryModule, IREE_ARRAYSIZE(kDuplicateEntryModule),
          IREE_ARRAYSIZE(entry_points), entry_points));
}

}  // namespace
}  // namespace iree::hal::vulkan
