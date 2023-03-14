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
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = NULL;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

TEST_P(executable_cache_test, CantPrepareUnknownFormat) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = NULL;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  EXPECT_FALSE(iree_hal_executable_cache_can_prepare_format(
      executable_cache, /*caching_mode=*/0, iree_make_cstring_view("FOO?")));

  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

TEST_P(executable_cache_test, PrepareExecutable) {
  iree_status_t loop_status = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache = NULL;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"),
      iree_loop_inline(&loop_status), &executable_cache));

  // Note: this layout must match the testdata executable.
  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {
          0,
          IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
      {
          1,
          IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_pipeline_layout_t* pipeline_layout;
  IREE_ASSERT_OK(iree_hal_pipeline_layout_create(
      device_, /*push_constants=*/0, /*set_layout_count=*/1,
      &descriptor_set_layout, &pipeline_layout));

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(get_test_executable_format());
  executable_params.executable_data = get_test_executable_data(
      iree_make_cstring_view("executable_cache_test.bin"));
  executable_params.pipeline_layout_count = 1;
  executable_params.pipeline_layouts = &pipeline_layout;

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, &executable));

  iree_hal_executable_release(executable);
  iree_hal_pipeline_layout_release(pipeline_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
  iree_hal_executable_cache_release(executable_cache);
  IREE_ASSERT_OK(loop_status);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_EXECUTABLE_CACHE_TEST_H_
