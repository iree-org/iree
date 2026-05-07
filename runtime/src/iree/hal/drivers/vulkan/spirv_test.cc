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

  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_TRUE(analysis.uses_physical_storage_buffer64_glsl450);
  EXPECT_EQ(1u, analysis.compute_entry_point_count);
  EXPECT_EQ(5u, analysis.compute_entry_point_name_storage_size);

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

TEST(SpirvTest, AnalyzesModuleWideFacts) {
  static constexpr uint32_t kMixedFactModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // Declares OpCapability PhysicalStorageBufferAddresses.
      0x00020011u,
      5347u,
      // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
      0x0003000eu,
      5348u,
      1u,
      // Declares OpDecorate %1 DescriptorSet 0.
      0x00040047u,
      1u,
      34u,
      0u,
      // Declares OpDecorate %1 Binding 2.
      0x00040047u,
      1u,
      33u,
      2u,
      // Declares OpVariable %3 in PushConstant storage class.
      0x0004003bu,
      2u,
      3u,
      9u,
      // Declares OpVariable %5 in StorageBuffer storage class.
      0x0004003bu,
      4u,
      5u,
      12u,
  };

  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kMixedFactModule, IREE_ARRAYSIZE(kMixedFactModule), &analysis));
  EXPECT_TRUE(analysis.uses_physical_storage_buffer64_glsl450);
  EXPECT_TRUE(analysis.has_physical_storage_buffer_addresses_capability);
  EXPECT_TRUE(analysis.has_descriptor_binding_decorations);
  EXPECT_EQ(1u, analysis.push_constant_variable_count);
  EXPECT_TRUE(analysis.has_descriptor_storage_class_variables);
  EXPECT_EQ(0u, analysis.compute_entry_point_count);
  EXPECT_EQ(0u, analysis.compute_entry_point_name_storage_size);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_TRUE(analysis.uses_physical_storage_buffer64_glsl450);
  EXPECT_TRUE(analysis.has_physical_storage_buffer_addresses_capability);
  EXPECT_FALSE(analysis.has_descriptor_binding_decorations);
  EXPECT_EQ(0u, analysis.push_constant_variable_count);
  EXPECT_FALSE(analysis.has_descriptor_storage_class_variables);
  EXPECT_EQ(1u, analysis.compute_entry_point_count);
  EXPECT_EQ(5u, analysis.compute_entry_point_name_storage_size);
}

TEST(SpirvTest, ParsesMultipleComputeEntryPoints) {
  static constexpr uint32_t kMultiEntryModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // Declares OpCapability PhysicalStorageBufferAddresses.
      0x00020011u,
      5347u,
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
      // Declares OpEntryPoint GLCompute %2 "aux".
      0x0004000fu,
      5u,
      2u,
      0x00787561u,
      // Declares OpExecutionMode %1 LocalSize 4 5 6.
      0x00060010u,
      1u,
      17u,
      4u,
      5u,
      6u,
      // Declares OpExecutionMode %2 LocalSize 7 8 9.
      0x00060010u,
      2u,
      17u,
      7u,
      8u,
      9u,
  };

  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kMultiEntryModule, IREE_ARRAYSIZE(kMultiEntryModule), &analysis));
  EXPECT_EQ(2u, analysis.compute_entry_point_count);
  EXPECT_EQ(9u, analysis.compute_entry_point_name_storage_size);

  iree_hal_vulkan_spirv_compute_entry_point_t entry_points[2] = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_parse_compute_entry_points(
      kMultiEntryModule, IREE_ARRAYSIZE(kMultiEntryModule),
      IREE_ARRAYSIZE(entry_points), entry_points));
  EXPECT_EQ(1u, entry_points[0].id);
  EXPECT_TRUE(iree_string_view_equal(entry_points[0].name, IREE_SV("main")));
  EXPECT_EQ(4u, entry_points[0].workgroup_size[0]);
  EXPECT_EQ(5u, entry_points[0].workgroup_size[1]);
  EXPECT_EQ(6u, entry_points[0].workgroup_size[2]);
  EXPECT_EQ(2u, entry_points[1].id);
  EXPECT_TRUE(iree_string_view_equal(entry_points[1].name, IREE_SV("aux")));
  EXPECT_EQ(7u, entry_points[1].workgroup_size[0]);
  EXPECT_EQ(8u, entry_points[1].workgroup_size[1]);
  EXPECT_EQ(9u, entry_points[1].workgroup_size[2]);
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

TEST(SpirvTest, RejectsTruncatedEntryPointOperands) {
  static constexpr uint32_t kTruncatedEntryPointModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpEntryPoint declares only ExecutionModel and EntryPoint id.
      0x0003000fu,
      5u,
      1u,
  };
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_analyze_module(
          kTruncatedEntryPointModule,
          IREE_ARRAYSIZE(kTruncatedEntryPointModule), &analysis));
}

TEST(SpirvTest, RejectsUnterminatedEntryPointNameDuringWorkgroupLookup) {
  static constexpr uint32_t kUnterminatedEntryPointModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // OpEntryPoint GLCompute %1 "main" without a trailing NUL.
      0x0004000fu,
      5u,
      1u,
      0x6e69616du,
      // OpExecutionMode %1 LocalSize 4 5 6.
      0x00060010u,
      1u,
      17u,
      4u,
      5u,
      6u,
  };
  bool entry_point_found = false;
  uint32_t workgroup_size[3] = {};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_parse_compute_workgroup_size(
          kUnterminatedEntryPointModule,
          IREE_ARRAYSIZE(kUnterminatedEntryPointModule), IREE_SV("main"),
          &entry_point_found, workgroup_size));
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
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kLogicalModule, IREE_ARRAYSIZE(kLogicalModule), &analysis));
  EXPECT_FALSE(analysis.uses_physical_storage_buffer64_glsl450);
}

TEST(SpirvTest, DetectsPhysicalStorageBufferAddressesCapability) {
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_TRUE(analysis.has_physical_storage_buffer_addresses_capability);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability,
      IREE_ARRAYSIZE(
          kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability),
      &analysis));
  EXPECT_FALSE(analysis.has_physical_storage_buffer_addresses_capability);
}

TEST(SpirvTest, VerifiesBdaModuleAbi) {
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_verify_bda_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
      IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_bda_module(
          kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
          IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT));

  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_bda_module(
          kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability,
          IREE_ARRAYSIZE(
              kComputeBdaModuleWithoutPhysicalStorageBufferAddressesCapability),
          IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_bda_module(
          kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
          (iree_hal_vulkan_spirv_bda_verification_flags_t)0x80000000u));
}

TEST(SpirvTest, VerifiesBdaEntryPoint) {
  uint32_t workgroup_size[3] = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_verify_bda_entry_point(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), IREE_SV("main"),
      IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE, workgroup_size));
  EXPECT_EQ(4u, workgroup_size[0]);
  EXPECT_EQ(5u, workgroup_size[1]);
  EXPECT_EQ(6u, workgroup_size[2]);

  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_bda_entry_point(
          kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule),
          IREE_SV("missing"), IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE,
          workgroup_size));
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
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kDescriptorDecoratedModule, IREE_ARRAYSIZE(kDescriptorDecoratedModule),
      &analysis));
  EXPECT_TRUE(analysis.has_descriptor_binding_decorations);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_FALSE(analysis.has_descriptor_binding_decorations);
}

TEST(SpirvTest, CountsPushConstantVariables) {
  static constexpr uint32_t kPushConstantModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // Declares OpVariable %3 in PushConstant storage class.
      0x0004003bu,
      2u,
      3u,
      9u,
  };
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kPushConstantModule, IREE_ARRAYSIZE(kPushConstantModule), &analysis));
  EXPECT_EQ(1u, analysis.push_constant_variable_count);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_EQ(0u, analysis.push_constant_variable_count);
}

TEST(SpirvTest, VerifiesBdaRootPushConstantLayout) {
  static constexpr uint32_t kBdaRootModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      16u,
      0u,
      // Declares OpTypeInt %1 64 0.
      0x00040015u,
      1u,
      64u,
      0u,
      // Declares OpTypeInt %2 32 0.
      0x00040015u,
      2u,
      32u,
      0u,
      // Declares OpTypeStruct %3 %1 %1 %2 %2 %2 %2.
      0x0008001eu,
      3u,
      1u,
      1u,
      2u,
      2u,
      2u,
      2u,
      // Declares OpTypePointer %4 PushConstant %3.
      0x00040020u,
      4u,
      9u,
      3u,
      // Declares OpVariable %5 in PushConstant storage class.
      0x0004003bu,
      4u,
      5u,
      9u,
      // Declares OpDecorate %3 Block.
      0x00030047u,
      3u,
      2u,
      // Declares BDA root member offsets.
      0x00050048u,
      3u,
      0u,
      35u,
      0u,
      0x00050048u,
      3u,
      1u,
      35u,
      8u,
      0x00050048u,
      3u,
      2u,
      35u,
      16u,
      0x00050048u,
      3u,
      3u,
      35u,
      20u,
      0x00050048u,
      3u,
      4u,
      35u,
      24u,
      0x00050048u,
      3u,
      5u,
      35u,
      28u,
  };
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
      kBdaRootModule, IREE_ARRAYSIZE(kBdaRootModule)));
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
          kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule)));
}

TEST(SpirvTest, DetectsDescriptorStorageClassVariables) {
  static constexpr uint32_t kDescriptorStorageVariableModule[] = {
      0x07230203u,
      0x00010600u,
      0u,
      8u,
      0u,
      // Declares OpVariable %3 in StorageBuffer storage class.
      0x0004003bu,
      2u,
      3u,
      12u,
  };
  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kDescriptorStorageVariableModule,
      IREE_ARRAYSIZE(kDescriptorStorageVariableModule), &analysis));
  EXPECT_TRUE(analysis.has_descriptor_storage_class_variables);

  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      kComputeBdaModule, IREE_ARRAYSIZE(kComputeBdaModule), &analysis));
  EXPECT_FALSE(analysis.has_descriptor_storage_class_variables);
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

  bool entry_point_found = false;
  uint32_t workgroup_size[3] = {};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_spirv_parse_compute_workgroup_size(
          kDuplicateEntryModule, IREE_ARRAYSIZE(kDuplicateEntryModule),
          IREE_SV("main"), &entry_point_found, workgroup_size));
}

}  // namespace
}  // namespace iree::hal::vulkan
