// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class ExecutableLayoutTest : public CtsTestBase {};

TEST_P(ExecutableLayoutTest, CreateWithNoLayouts) {
  iree_hal_executable_layout_t* executable_layout;
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*push_constants=*/0, /*set_layout_count=*/0, NULL,
      &executable_layout));

  iree_hal_executable_layout_release(executable_layout);
}

TEST_P(ExecutableLayoutTest, CreateWithPushConstants) {
  iree_hal_executable_layout_t* executable_layout;
  // Note: The Vulkan maxPushConstantsSize limit must be at least 128 bytes:
  // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#limits-minmax
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*push_constants=*/5, /*set_layout_count=*/0, NULL,
      &executable_layout));

  iree_hal_executable_layout_release(executable_layout);
}

TEST_P(ExecutableLayoutTest, CreateWithOneLayout) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/1, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));

  iree_hal_executable_layout_t* executable_layout;
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*push_constants=*/0, /*set_layout_count=*/1,
      &descriptor_set_layout, &executable_layout));

  iree_hal_executable_layout_release(executable_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(ExecutableLayoutTest, CreateWithTwoLayouts) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layouts[2];
  iree_hal_descriptor_set_layout_binding_t layout_bindings_0[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/1, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(layout_bindings_0), layout_bindings_0,
      &descriptor_set_layouts[0]));

  iree_hal_descriptor_set_layout_binding_t layout_bindings_1[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/1, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/2, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(layout_bindings_1), layout_bindings_1,
      &descriptor_set_layouts[1]));

  iree_hal_executable_layout_t* executable_layout;
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*push_constants=*/0, IREE_ARRAYSIZE(descriptor_set_layouts),
      descriptor_set_layouts, &executable_layout));

  iree_hal_executable_layout_release(executable_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layouts[0]);
  iree_hal_descriptor_set_layout_release(descriptor_set_layouts[1]);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, ExecutableLayoutTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
