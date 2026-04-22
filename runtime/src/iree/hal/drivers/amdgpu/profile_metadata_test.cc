// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_metadata.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class ProfileMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amdgpu_profile_metadata_initialize(iree_allocator_system(),
                                                &registry_);
  }

  void TearDown() override {
    iree_hal_amdgpu_profile_metadata_deinitialize(&registry_);
  }

  iree_hal_executable_export_info_t MakeExportInfo() {
    iree_hal_executable_export_info_t export_info = {};
    export_info.name = IREE_SV("test_dispatch");
    export_info.constant_count = 3;
    export_info.binding_count = 2;
    export_info.workgroup_size[0] = 8;
    export_info.workgroup_size[1] = 4;
    export_info.workgroup_size[2] = 1;
    return export_info;
  }

  iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs() {
    iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
    kernel_args.workgroup_size[0] = 8;
    kernel_args.workgroup_size[1] = 4;
    kernel_args.workgroup_size[2] = 1;
    kernel_args.constant_count = 3;
    kernel_args.binding_count = 2;
    return kernel_args;
  }

  iree_hal_amdgpu_profile_metadata_registry_t registry_;
};

TEST_F(ProfileMetadataTest, HashCodeObjectGolden) {
  const uint8_t code_object_data[] = {
      0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00,
      'I',  'R', 'E', 'E', 0x00, 0x10, 0x80, 0xff,
  };

  uint64_t code_object_hash[2] = {};
  iree_hal_amdgpu_profile_metadata_hash_code_object(
      iree_make_const_byte_span(code_object_data, sizeof(code_object_data)),
      code_object_hash);

  EXPECT_EQ(code_object_hash[0], 0xc928aa6b62b629a9ull);
  EXPECT_EQ(code_object_hash[1], 0xd1a52f9c3082fba5ull);
}

TEST_F(ProfileMetadataTest, RegisterExecutableRecordsOnlyIdentity) {
  iree_hal_executable_export_info_t export_info = MakeExportInfo();
  iree_host_size_t export_parameter_offsets[] = {0, 0};
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs();
  uint64_t code_object_hash[2] = {0x1111111111111111ull, 0x2222222222222222ull};

  uint64_t executable_id = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_metadata_register_executable(
      &registry_, /*export_count=*/1, &export_info, export_parameter_offsets,
      code_object_hash, &kernel_args, &executable_id));

  EXPECT_EQ(executable_id, 1u);
  ASSERT_EQ(registry_.executable_record_count, 1u);
  EXPECT_EQ(registry_.executable_records[0].executable_id, executable_id);
  EXPECT_EQ(registry_.executable_records[0].export_count, 1u);
  EXPECT_NE(registry_.executable_export_record_data_length, 0u);
  EXPECT_EQ(registry_.executable_code_object_record_data_length, 0u);
  EXPECT_EQ(registry_.executable_code_object_load_record_count, 0u);
}

TEST_F(ProfileMetadataTest, RegisterExecutableComputesStablePipelineHash) {
  iree_hal_executable_export_info_t export_info = MakeExportInfo();
  iree_host_size_t export_parameter_offsets[] = {0, 3};
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs();
  uint64_t code_object_hash[2] = {0x0706050403020100ull, 0x1716151413121110ull};

  uint64_t executable_id = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_metadata_register_executable(
      &registry_, /*export_count=*/1, &export_info, export_parameter_offsets,
      code_object_hash, &kernel_args, &executable_id));

  ASSERT_EQ(registry_.executable_export_record_data_length,
            sizeof(iree_hal_profile_executable_export_record_t) +
                export_info.name.size);
  const auto* export_record =
      reinterpret_cast<const iree_hal_profile_executable_export_record_t*>(
          registry_.executable_export_record_data);
  ASSERT_NE(export_record, nullptr);
  EXPECT_EQ(export_record->flags,
            IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH);
  EXPECT_EQ(export_record->executable_id, executable_id);
  EXPECT_EQ(export_record->export_ordinal, 0u);
  EXPECT_EQ(export_record->constant_count, 3u);
  EXPECT_EQ(export_record->binding_count, 2u);
  EXPECT_EQ(export_record->parameter_count, 3u);
  EXPECT_EQ(export_record->workgroup_size[0], 8u);
  EXPECT_EQ(export_record->workgroup_size[1], 4u);
  EXPECT_EQ(export_record->workgroup_size[2], 1u);
  EXPECT_EQ(export_record->pipeline_hash[0], 0x12dbd8b44277f553ull);
  EXPECT_EQ(export_record->pipeline_hash[1], 0x873c5d1c5596dce4ull);
}

TEST_F(ProfileMetadataTest, RegisterExecutableArtifactsAttachToIdentity) {
  iree_hal_executable_export_info_t export_info = MakeExportInfo();
  iree_host_size_t export_parameter_offsets[] = {0, 0};
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs();
  uint64_t code_object_hash[2] = {0x1111111111111111ull, 0x2222222222222222ull};

  uint64_t executable_id = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_metadata_register_executable(
      &registry_, /*export_count=*/1, &export_info, export_parameter_offsets,
      code_object_hash, &kernel_args, &executable_id));

  const uint8_t code_object_data[] = {0x7f, 'E', 'L', 'F', 0x01};
  const iree_hal_amdgpu_profile_code_object_load_info_t load_infos[] = {
      {
          .physical_device_ordinal = 0,
          .load_delta = 0x1000,
          .load_size = 0x2000,
      },
      {
          .physical_device_ordinal = 1,
          .load_delta = 0x3000,
          .load_size = 0x4000,
      },
  };
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_metadata_register_executable_artifacts(
      &registry_, executable_id,
      iree_make_const_byte_span(code_object_data, sizeof(code_object_data)),
      code_object_hash, IREE_ARRAYSIZE(load_infos), load_infos));

  EXPECT_NE(registry_.executable_code_object_record_data_length, 0u);
  EXPECT_EQ(registry_.executable_code_object_load_record_count, 2u);

  iree_hal_profile_executable_code_object_load_record_t load_record;
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_metadata_lookup_code_object_load(
      &registry_, executable_id, /*physical_device_ordinal=*/1, &load_record));
  EXPECT_EQ(load_record.executable_id, executable_id);
  EXPECT_EQ(load_record.code_object_id, executable_id);
  EXPECT_EQ(load_record.load_delta, 0x3000);
  EXPECT_EQ(load_record.load_size, 0x4000u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_hal_amdgpu_profile_metadata_register_executable_artifacts(
          &registry_, executable_id,
          iree_make_const_byte_span(code_object_data, sizeof(code_object_data)),
          code_object_hash, IREE_ARRAYSIZE(load_infos), load_infos));
}

TEST_F(ProfileMetadataTest, RegisterExecutableArtifactsRequiresIdentity) {
  const uint8_t code_object_data[] = {0x7f, 'E', 'L', 'F', 0x01};
  uint64_t code_object_hash[2] = {0x1111111111111111ull, 0x2222222222222222ull};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_hal_amdgpu_profile_metadata_register_executable_artifacts(
          &registry_, /*executable_id=*/42,
          iree_make_const_byte_span(code_object_data, sizeof(code_object_data)),
          code_object_hash, /*code_object_load_info_count=*/0,
          /*code_object_load_infos=*/nullptr));
}

}  // namespace
}  // namespace iree::hal::amdgpu
