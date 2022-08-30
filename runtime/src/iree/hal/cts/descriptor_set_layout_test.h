// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_DESCRIPTOR_SET_LAYOUT_TEST_H_
#define IREE_HAL_CTS_DESCRIPTOR_SET_LAYOUT_TEST_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class descriptor_set_layout_test : public CtsTestBase {};

// Note: bindingCount == 0 is valid in VkDescriptorSetLayoutCreateInfo:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSetLayoutCreateInfo.html
TEST_P(descriptor_set_layout_test, CreateWithNoBindings) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      /*binding_count=*/0,
      /*bindings=*/NULL, &descriptor_set_layout));
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(descriptor_set_layout_test, CreateWithOneBinding) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {
          /*binding=*/0,
          /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          /*flags=*/IREE_HAL_DESCRIPTOR_FLAG_NONE,
      },
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(descriptor_set_layout_test, CreateWithTwoBindings) {
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
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(descriptor_set_layout_test, CreateWithPushDescriptorType) {
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
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_DESCRIPTOR_SET_LAYOUT_TEST_H_
