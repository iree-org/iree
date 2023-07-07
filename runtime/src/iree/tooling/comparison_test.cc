// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/comparison.h"

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

using ::testing::HasSubstr;

static void ParseToVariantList(iree_hal_allocator_t* device_allocator,
                               iree::span<const std::string> input_strings,
                               iree_allocator_t host_allocator,
                               iree_vm_list_t** out_list) {
  std::vector<iree_string_view_t> input_string_views(input_strings.size());
  for (size_t i = 0; i < input_strings.size(); ++i) {
    input_string_views[i].data = input_strings[i].data();
    input_string_views[i].size = input_strings[i].size();
  }
  IREE_CHECK_OK(iree_tooling_parse_to_variant_list(
      device_allocator, input_string_views.data(), input_string_views.size(),
      host_allocator, out_list));
}

class ComparisonTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           host_allocator_, &instance_));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(instance_));
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("heap"), host_allocator_, host_allocator_, &device_allocator_));
  }

  virtual void TearDown() {
    iree_hal_allocator_release(device_allocator_);
    iree_vm_instance_release(instance_);
  }

  bool ParseAndCompareVariantLists(
      iree::span<const std::string> expected_strings,
      iree::span<const std::string> actual_strings, std::string* out_string) {
    vm::ref<iree_vm_list_t> expected_list;
    ParseToVariantList(device_allocator_, expected_strings, host_allocator_,
                       &expected_list);

    vm::ref<iree_vm_list_t> actual_list;
    ParseToVariantList(device_allocator_, actual_strings, host_allocator_,
                       &actual_list);

    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator_, &builder);
    bool all_match = iree_tooling_compare_variant_lists_and_append(
        expected_list.get(), actual_list.get(), host_allocator_, &builder);
    out_string->assign(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);

    return all_match;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ComparisonTest, CompareEqualLists) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  auto buf_strings = std::vector<std::string>{buf_string1, buf_string2};
  std::string result;
  EXPECT_TRUE(ParseAndCompareVariantLists(buf_strings, buf_strings, &result));
  EXPECT_EQ(result, "");
}

TEST_F(ComparisonTest, CompareListsWithIgnored) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 999][4 5 6]";
  std::string buf_string2_ignored = "(ignored)";
  auto actual_strings = std::vector<std::string>{buf_string1, buf_string2};
  auto expected_strings =
      std::vector<std::string>{buf_string1, buf_string2_ignored};
  std::string result;
  EXPECT_TRUE(
      ParseAndCompareVariantLists(expected_strings, actual_strings, &result));
  EXPECT_EQ(result, "");
}

TEST_F(ComparisonTest, CompareTruncatedLists) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  auto actual_strings = std::vector<std::string>{buf_string1, buf_string2};
  auto expected_strings = std::vector<std::string>{buf_string1};
  std::string result;
  EXPECT_FALSE(
      ParseAndCompareVariantLists(expected_strings, actual_strings, &result));
  EXPECT_THAT(result, HasSubstr("expected 1 list elements but 2 provided"));
}

TEST_F(ComparisonTest, CompareDifferingLists) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 999][4 5 6]";
  std::string buf_string2_good = "2x3xf64=[1 2 3][4 5 6]";
  auto actual_strings = std::vector<std::string>{buf_string1, buf_string2};
  auto expected_strings =
      std::vector<std::string>{buf_string1, buf_string2_good};
  std::string result;
  EXPECT_FALSE(
      ParseAndCompareVariantLists(expected_strings, actual_strings, &result));
  EXPECT_THAT(
      result,
      HasSubstr("element at index 2 (999) does not match the expected (3)"));
}

TEST_F(ComparisonTest, CompareListsWithDifferingTypes) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "123";
  std::string buf_string2_good = "2x3xf64=[1 2 3][4 5 6]";
  auto actual_strings = std::vector<std::string>{buf_string1, buf_string2};
  auto expected_strings =
      std::vector<std::string>{buf_string1, buf_string2_good};
  std::string result;
  EXPECT_FALSE(
      ParseAndCompareVariantLists(expected_strings, actual_strings, &result));
  EXPECT_THAT(result, HasSubstr("variant types mismatch"));
}

}  // namespace
}  // namespace iree
