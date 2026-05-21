// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libaqlprofile.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Tests that we can find, load, query, and unload aqlprofile. If HSA or
// aqlprofile cannot be found then we skip the test so that it doesn't fail on
// machines without ROCm profiling libraries installed.
TEST(LibAqlprofileTest, Load) {
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
      iree_allocator_system(), &libhsa);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    GTEST_SKIP() << "HSA not available, skipping test";
  }

  iree_hal_amdgpu_libaqlprofile_t libaqlprofile;
  status = iree_hal_amdgpu_libaqlprofile_initialize(
      &libhsa, iree_string_view_list_empty(), iree_allocator_system(),
      &libaqlprofile);
  if (iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
    GTEST_SKIP() << "aqlprofile not available, skipping test";
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }
  IREE_ASSERT_OK(status);

  EXPECT_NE(libaqlprofile.version.major, 0u);
  EXPECT_NE(libaqlprofile.aqlprofile_get_version, nullptr);
  EXPECT_NE(libaqlprofile.aqlprofile_register_agent_info, nullptr);
  EXPECT_NE(libaqlprofile.aqlprofile_validate_pmc_event, nullptr);
  EXPECT_NE(libaqlprofile.aqlprofile_pmc_create_packets, nullptr);
  EXPECT_NE(libaqlprofile.aqlprofile_pmc_delete_packets, nullptr);
  EXPECT_NE(libaqlprofile.aqlprofile_pmc_iterate_data, nullptr);

  iree_hal_amdgpu_libaqlprofile_deinitialize(&libaqlprofile);
  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}

}  // namespace
}  // namespace iree::hal::amdgpu
