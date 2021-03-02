// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class DescriptorSetLayoutTest : public CtsTestBase {};

// Note: bindingCount == 0 is valid in VkDescriptorSetLayoutCreateInfo:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSetLayoutCreateInfo.html
TEST_P(DescriptorSetLayoutTest, CreateWithNoBindings) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      /*binding_count=*/0,
      /*bindings=*/NULL, &descriptor_set_layout));
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(DescriptorSetLayoutTest, CreateWithOneBinding) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(DescriptorSetLayoutTest, CreateWithTwoBindings) {
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
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

TEST_P(DescriptorSetLayoutTest, CreateWithPushDescriptorType) {
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/1, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_PUSH_ONLY,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, DescriptorSetLayoutTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
