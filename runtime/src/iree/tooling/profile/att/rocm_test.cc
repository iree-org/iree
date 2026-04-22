// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/rocm.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AttRocmTest, ExplicitPathTakesPrecedence) {
  iree_string_view_t path =
      iree_profile_att_rocm_library_path_or_env(IREE_SV("/opt/rocm/lib"));
  EXPECT_TRUE(iree_string_view_equal(path, IREE_SV("/opt/rocm/lib")));
}

TEST(AttRocmTest, ResolvesExplicitLibraryDirectory) {
  char* directory = nullptr;
  IREE_ASSERT_OK(iree_profile_att_rocm_resolve_library_dir(
      /*symbol=*/nullptr, IREE_SV("/opt/rocm/lib/libamd_comgr.so"),
      iree_allocator_system(), &directory));
  EXPECT_STREQ("/opt/rocm/lib", directory);
  iree_allocator_free(iree_allocator_system(), directory);
}

}  // namespace
