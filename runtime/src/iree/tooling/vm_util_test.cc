// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/vm_util.h"

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

static Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                                 iree::span<const std::string> input_strings,
                                 iree_allocator_t host_allocator,
                                 iree_vm_list_t** out_list) {
  std::vector<iree_string_view_t> input_string_views(input_strings.size());
  for (size_t i = 0; i < input_strings.size(); ++i) {
    input_string_views[i].data = input_strings[i].data();
    input_string_views[i].size = input_strings[i].size();
  }
  return iree_tooling_parse_to_variant_list(
      device_allocator, input_string_views.data(), input_string_views.size(),
      host_allocator, out_list);
}

static Status PrintVariantList(iree_vm_list_t* variant_list,
                               std::string* out_string) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_RETURN_IF_ERROR(iree_tooling_append_variant_list_lines(
      IREE_SV("result"), variant_list, /*max_element_count=*/1024, &builder));
  out_string->assign(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

class VmUtilTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_ASSERT_OK(
        iree_vm_instance_create(iree_allocator_system(), &instance_));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(instance_));
    iree_status_t status = iree_hal_create_device(
        iree_hal_available_driver_registry(), IREE_SV("local-sync"),
        iree_allocator_system(), &device_);
    if (iree_status_is_not_found(status)) {
      fprintf(stderr, "Skipping test as 'local-sync' driver was not found:\n");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP();
    }
    allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() {
    iree_hal_device_release(device_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(VmUtilTest, ParsePrintBuffer) {
  std::string buf_string = "&2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(
      ParseToVariantList(allocator_, std::vector<std::string>{buf_string},
                         iree_vm_instance_allocator(instance_), &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer\n") + "(no printer)" + "\n");
}

TEST_F(VmUtilTest, ParsePrintBufferView) {
  std::string buf_string = "2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(
      ParseToVariantList(allocator_, std::vector<std::string>{buf_string},
                         iree_vm_instance_allocator(instance_), &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintScalar) {
  std::string input_string = "42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(
      ParseToVariantList(allocator_, std::vector<std::string>{input_string},
                         iree_vm_instance_allocator(instance_), &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result, std::string("result[0]: i32=") + input_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintRank0BufferView) {
  std::string buf_string = "i32=42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(
      ParseToVariantList(allocator_, std::vector<std::string>{buf_string},
                         iree_vm_instance_allocator(instance_), &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintMultipleBufferViews) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      allocator_, std::vector<std::string>{buf_string1, buf_string2},
      iree_vm_instance_allocator(instance_), &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result, std::string("result[0]: hal.buffer_view\n") + buf_string1 +
                        "\nresult[1]: hal.buffer_view\n" + buf_string2 + "\n");
}

}  // namespace
}  // namespace iree
