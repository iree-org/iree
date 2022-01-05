// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_EXECUTABLE_CACHE_TEST_H_
#define IREE_HAL_CTS_EXECUTABLE_CACHE_TEST_H_

#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class executable_cache_test : public CtsTestBase {};

TEST_P(executable_cache_test, Create) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  iree_hal_executable_cache_release(executable_cache);
}

TEST_P(executable_cache_test, CantPrepareUnknownFormat) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("FOO?")));

  iree_hal_executable_cache_release(executable_cache);
}

TEST_P(executable_cache_test, PrepareExecutable) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  // Note: this layout must match the testdata executable.
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {0, IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER},
      {1, IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_executable_layout_t* executable_layout;
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*push_constants=*/0, /*set_layout_count=*/1,
      &descriptor_set_layout, &executable_layout));

  iree_hal_executable_spec_t executable_spec;
  executable_spec.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_spec.executable_format =
      iree_make_cstring_view(get_test_executable_format());
  executable_spec.executable_data = get_test_executable_data(
      iree_make_cstring_view("executable_cache_test.bin"));
  executable_spec.executable_layout_count = 1;
  executable_spec.executable_layouts = &executable_layout;

  iree_hal_executable_t* executable;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_spec, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_layout_release(executable_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
  iree_hal_executable_cache_release(executable_cache);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_EXECUTABLE_CACHE_TEST_H_
