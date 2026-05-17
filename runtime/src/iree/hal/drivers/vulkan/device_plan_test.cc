// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_plan.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

class PhysicalDeviceSnapshotBuilder {
 public:
  PhysicalDeviceSnapshotBuilder() {
    std::memset(&snapshot_, 0, sizeof(snapshot_));
    snapshot_.properties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    snapshot_.properties2.properties.apiVersion = VK_API_VERSION_1_3;
    snapshot_.features12.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    snapshot_.features12.timelineSemaphore = VK_TRUE;
    snapshot_.features12.scalarBlockLayout = VK_TRUE;
    snapshot_.features13.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    snapshot_.features13.synchronization2 = VK_TRUE;
  }

  void AddQueueFamily(VkQueueFlags flags, uint32_t queue_count,
                      uint32_t timestamp_valid_bits = 64) {
    VkQueueFamilyProperties2 queue_family;
    std::memset(&queue_family, 0, sizeof(queue_family));
    queue_family.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    queue_family.queueFamilyProperties.queueFlags = flags;
    queue_family.queueFamilyProperties.queueCount = queue_count;
    queue_family.queueFamilyProperties.timestampValidBits =
        timestamp_valid_bits;
    queue_families_.push_back(queue_family);
    snapshot_.queue_family_count =
        static_cast<uint32_t>(queue_families_.size());
    snapshot_.queue_families = queue_families_.data();
  }

  void EnableSparseBinding() {
    snapshot_.features2.features.sparseBinding = VK_TRUE;
  }

  void EnableSparseResidencyAliased() {
    snapshot_.features2.features.sparseResidencyBuffer = VK_TRUE;
    snapshot_.features2.features.sparseResidencyAliased = VK_TRUE;
  }

  void EnableBufferDeviceAddress() {
    snapshot_.features12.bufferDeviceAddress = VK_TRUE;
  }

  void EnableScalarShaderFeatures() {
    snapshot_.features12.storageBuffer8BitAccess = VK_TRUE;
    snapshot_.features12.shaderFloat16 = VK_TRUE;
    snapshot_.features2.features.shaderFloat64 = VK_TRUE;
    snapshot_.features12.shaderInt8 = VK_TRUE;
    snapshot_.features2.features.shaderInt16 = VK_TRUE;
    snapshot_.features2.features.shaderInt64 = VK_TRUE;
    snapshot_.features13.shaderIntegerDotProduct = VK_TRUE;
    snapshot_.features12.vulkanMemoryModel = VK_TRUE;
    snapshot_.features12.vulkanMemoryModelDeviceScope = VK_TRUE;
  }

  void EnableCooperativeMatrixExtension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX;
  }

  void EnableCooperativeMatrix() {
    EnableCooperativeMatrixExtension();
    snapshot_.cooperative_matrix_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    snapshot_.cooperative_matrix_features.cooperativeMatrix = VK_TRUE;
    snapshot_.cooperative_matrix_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    snapshot_.cooperative_matrix_properties.cooperativeMatrixSupportedStages =
        VK_SHADER_STAGE_COMPUTE_BIT;
  }

  const iree_hal_vulkan_physical_device_snapshot_t* snapshot() const {
    return &snapshot_;
  }

 private:
  // Snapshot whose pointer fields reference this builder's storage.
  iree_hal_vulkan_physical_device_snapshot_t snapshot_;

  // Backing storage for snapshot_.queue_families.
  std::vector<VkQueueFamilyProperties2> queue_families_;
};

static iree_hal_vulkan_device_options_t DefaultDeviceOptions() {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  return options;
}

static iree_hal_vulkan_external_device_params_t DefaultExternalParams() {
  iree_hal_vulkan_external_device_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.enabled_features = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE;
  return params;
}

static bool PlanContainsExtension(const iree_hal_vulkan_device_plan_t& plan,
                                  const char* extension_name) {
  for (uint32_t i = 0; i < plan.enabled_extension_count; ++i) {
    if (std::strcmp(plan.enabled_extension_names[i], extension_name) == 0) {
      return true;
    }
  }
  return false;
}

TEST(DevicePlanTest, OwnedCreatePrefersDedicatedComputeFamily) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 2);
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.AddQueueFamily(VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  options.flags = IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_NONE, &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_NONE, plan.request_flags);
  EXPECT_EQ(1u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(2u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR,
            plan.enabled_dispatch_abis);
  EXPECT_EQ(2u, plan.queue_create_info_count);
}

TEST(DevicePlanTest,
     OwnedCreateUsesSecondQueueWhenTransferSharesComputeFamily) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 2);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_NONE, &plan));

  EXPECT_EQ(0u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(1u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(1u, plan.queue_create_info_count);
  EXPECT_EQ(2u, plan.queue_create_infos[0].queueCount);
}

TEST(DevicePlanTest, OwnedCreateKeepsSparseBindingSeparateFromSparseResidency) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT,
                         2);
  builder.EnableSparseBinding();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING, &plan));

  EXPECT_TRUE(plan.enabled_features2.features.sparseBinding);
  EXPECT_FALSE(plan.enabled_features2.features.sparseResidencyBuffer);
  EXPECT_FALSE(plan.enabled_features2.features.sparseResidencyAliased);
  EXPECT_TRUE(iree_hal_vulkan_queue_assignment_has_sparse_binding(
      &plan.queue_assignment));
}

TEST(DevicePlanTest, OwnedCreateRequiresCompleteSparseResidencyRequest) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT,
                         2);
  builder.EnableSparseBinding();
  builder.EnableSparseResidencyAliased();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  const iree_hal_vulkan_features_t sparse_residency_without_sparse_binding =
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED &
      ~IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          sparse_residency_without_sparse_binding, &plan));
}

TEST(DevicePlanTest, OwnedCreateRequiresBdaForBdaOnlyDispatch) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  options.dispatch_abis = IREE_HAL_VULKAN_DISPATCH_ABI_BDA;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES, &plan));
}

TEST(DevicePlanTest, OwnedCreateEnablesBdaDispatchWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableBufferDeviceAddress();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES, &plan));

  EXPECT_TRUE(plan.enabled_features12.bufferDeviceAddress);
  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_ALL_RECOGNIZED,
            plan.enabled_dispatch_abis);
}

TEST(DevicePlanTest, OwnedCreateReportsAvailableScalarShaderFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableScalarShaderFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_NONE, &plan));

  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features,
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT));
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features,
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
  EXPECT_TRUE(plan.enabled_features12.storageBuffer8BitAccess);
  EXPECT_TRUE(plan.enabled_features12.shaderFloat16);
  EXPECT_TRUE(plan.enabled_features2.features.shaderFloat64);
  EXPECT_TRUE(plan.enabled_features12.shaderInt8);
  EXPECT_TRUE(plan.enabled_features2.features.shaderInt16);
  EXPECT_TRUE(plan.enabled_features2.features.shaderInt64);
  EXPECT_TRUE(plan.enabled_features13.shaderIntegerDotProduct);
  EXPECT_TRUE(plan.enabled_features12.vulkanMemoryModel);
  EXPECT_TRUE(plan.enabled_features12.vulkanMemoryModelDeviceScope);
}

TEST(DevicePlanTest, OwnedCreateRejectsRequestedUnavailableReportedFeature) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16, &plan));
}

TEST(DevicePlanTest, OwnedCreateEnablesCooperativeMatrixWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrix();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_NONE, &plan));

  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_extensions,
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX));
  EXPECT_TRUE(
      PlanContainsExtension(plan, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME));
  EXPECT_TRUE(plan.enabled_cooperative_matrix_features.cooperativeMatrix);

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_features2, create_info.pNext);
  EXPECT_EQ(&copied_plan.enabled_features12,
            copied_plan.enabled_features2.pNext);
  EXPECT_EQ(&copied_plan.enabled_features13,
            copied_plan.enabled_features12.pNext);
  EXPECT_EQ(&copied_plan.enabled_cooperative_matrix_features,
            copied_plan.enabled_features13.pNext);
}

TEST(DevicePlanTest, OwnedCreateCarriesRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS,
      IREE_HAL_VULKAN_FEATURE_NONE, &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, plan.request_flags);
}

TEST(DevicePlanTest, OwnedCreateRejectsUnknownRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_create(
                            builder.snapshot(), &options,
                            IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1,
                            IREE_HAL_VULKAN_FEATURE_NONE, &plan));
}

TEST(DevicePlanTest, WrapRejectsRequestFlagsInEnabledFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features |= IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapInfersTransferFromComputeWhenSupported) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_EQ(0u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(1ull << 1, plan.queue_assignment.transfer.affinity);
  EXPECT_EQ(IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR,
            plan.enabled_dispatch_abis);
}

TEST(DevicePlanTest, WrapCarriesRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, plan.request_flags);
}

TEST(DevicePlanTest, WrapRejectsCooperativeMatrixWithoutEnabledExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrix();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsCooperativeMatrixWhenFeatureIsUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrixExtension();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsUnknownRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRequiresTransferQueueWhenComputeCannotTransfer) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  iree_hal_vulkan_device_plan_t plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, MakeCreateInfoRefreshesSelfReferencesAfterCopy) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 2);
  builder.AddQueueFamily(VK_QUEUE_SPARSE_BINDING_BIT, 1);
  builder.EnableSparseBinding();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  iree_hal_vulkan_device_plan_t plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING, &plan));

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_features2, create_info.pNext);
  EXPECT_EQ(&copied_plan.enabled_features12,
            copied_plan.enabled_features2.pNext);
  EXPECT_EQ(&copied_plan.enabled_features13,
            copied_plan.enabled_features12.pNext);
  EXPECT_EQ(copied_plan.queue_create_infos, create_info.pQueueCreateInfos);
  EXPECT_EQ(&copied_plan.queue_priorities[0],
            copied_plan.queue_create_infos[0].pQueuePriorities);
  EXPECT_EQ(&copied_plan.queue_priorities[2],
            copied_plan.queue_create_infos[1].pQueuePriorities);
}

}  // namespace
}  // namespace iree::hal::vulkan
