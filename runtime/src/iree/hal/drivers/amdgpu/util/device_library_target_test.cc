// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library_target.h"

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static std::vector<std::string> CandidateValues(const char* isa_name) {
  iree_hal_amdgpu_device_library_target_candidate_list_t candidates = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_device_library_target_candidates_from_isa(
      iree_make_cstring_view(isa_name), &candidates));
  std::vector<std::string> values;
  values.reserve(candidates.count);
  for (iree_host_size_t i = 0; i < candidates.count; ++i) {
    values.emplace_back(candidates.values[i].value.data,
                        candidates.values[i].value.size);
  }
  return values;
}

TEST(DeviceLibraryTargetTest,
     PreservesFeatureBearingCandidatesBeforeFallbacks) {
  const auto values =
      CandidateValues("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-");

  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(values[0], "gfx942:sramecc+:xnack-");
  EXPECT_EQ(values[1], "gfx942");
  EXPECT_EQ(values[2], "gfx9-4-generic:sramecc+:xnack-");
  EXPECT_EQ(values[3], "gfx9-4-generic");
}

TEST(DeviceLibraryTargetTest, MapsGfx12_5TargetsToGenericFamily) {
  const auto values = CandidateValues("amdgcn-amd-amdhsa--gfx1250");

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], "gfx1250");
  EXPECT_EQ(values[1], "gfx12-5-generic");
}

TEST(DeviceLibraryTargetTest, MatchesOnlyWholeFileArchSegments) {
  EXPECT_TRUE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx9-4-generic.so"), IREE_SV("gfx9-4-generic")));
  EXPECT_TRUE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942.debug.so"), IREE_SV("gfx942")));

  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942x.so"), IREE_SV("gfx942")));
  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx9-4-generic.so"), IREE_SV("gfx9-4-generic:sramecc+:xnack-")));
  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942.so"), IREE_SV("")));
}

}  // namespace
}  // namespace iree::hal::amdgpu
