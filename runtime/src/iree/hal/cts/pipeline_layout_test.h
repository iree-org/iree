// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_PIPELINE_LAYOUT_TEST_H_
#define IREE_HAL_CTS_PIPELINE_LAYOUT_TEST_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class pipeline_layout_test : public CtsTestBase {};

TEST_P(pipeline_layout_test, CreateWithNoLayouts) {
  iree_hal_pipeline_layout_t* pipeline_layout = NULL;
  IREE_ASSERT_OK(iree_hal_pipeline_layout_create(device_, /*push_constants=*/0,
                                                 /*set_layout_count=*/0, NULL,
                                                 &pipeline_layout));

  iree_hal_pipeline_layout_release(pipeline_layout);
}

TEST_P(pipeline_layout_test, CreateWithPushConstants) {
  iree_hal_pipeline_layout_t* pipeline_layout = NULL;
  // Note: The Vulkan maxPushConstantsSize limit must be at least 128 bytes:
  // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#limits-minmax
  IREE_ASSERT_OK(iree_hal_pipeline_layout_create(device_, /*push_constants=*/5,
                                                 /*set_layout_count=*/0, NULL,
                                                 &pipeline_layout));

  iree_hal_pipeline_layout_release(pipeline_layout);
}

TEST_P(pipeline_layout_test, CreateWithOneLayout) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {
          /*binding=*/0,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
      {
          /*binding=*/1,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));

  iree_hal_pipeline_layout_t* pipeline_layout = NULL;
  IREE_ASSERT_OK(iree_hal_pipeline_layout_create(
      device_, /*push_constants=*/0, /*set_layout_count=*/1,
      &descriptor_set_layout, &pipeline_layout));

  iree_hal_pipeline_layout_release(pipeline_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(pipeline_layout_test, CreateWithTwoLayouts) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layouts[2] = {NULL};
  iree_hal_descriptor_set_layout_binding_t layout_bindings_0[] = {
      {
          /*binding=*/0,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
      {
          /*binding=*/1,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      IREE_ARRAYSIZE(layout_bindings_0), layout_bindings_0,
      &descriptor_set_layouts[0]));

  iree_hal_descriptor_set_layout_binding_t layout_bindings_1[] = {
      {
          /*binding=*/0,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
      {
          /*binding=*/1,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
      {
          /*binding=*/2,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      IREE_ARRAYSIZE(layout_bindings_1), layout_bindings_1,
      &descriptor_set_layouts[1]));

  iree_hal_pipeline_layout_t* pipeline_layout = NULL;
  IREE_ASSERT_OK(iree_hal_pipeline_layout_create(
      device_, /*push_constants=*/0, IREE_ARRAYSIZE(descriptor_set_layouts),
      descriptor_set_layouts, &pipeline_layout));

  iree_hal_pipeline_layout_release(pipeline_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layouts[0]);
  iree_hal_descriptor_set_layout_release(descriptor_set_layouts[1]);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_PIPELINE_LAYOUT_TEST_H_
