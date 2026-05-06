// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Tests that we can find, load, and unload HSA.
// In ASAN builds it tests that we don't leak the library (though ROCR itself
// leaks a bunch). If the library cannot be found then we skip the test so that
// it doesn't fail on machines without HSA installed.
TEST(LibHSATest, Load) {
  iree_hal_amdgpu_libhsa_t libhsa;
  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
      iree_allocator_system(), &libhsa);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    GTEST_SKIP() << "HSA not available, skipping tests";
  }

  // Ensure resolved symbols are callable without perturbing the HSA runtime
  // lifetime beyond the one owned by libhsa.
  uint16_t version_major = 0;
  IREE_ASSERT_OK(iree_hsa_system_get_info(
      IREE_LIBHSA(&libhsa), HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major));
  EXPECT_NE(version_major, 0u);

  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
}

}  // namespace
}  // namespace iree::hal::amdgpu
