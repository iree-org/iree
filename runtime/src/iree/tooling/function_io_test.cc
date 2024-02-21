// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/function_io.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/vec_stream.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

struct FunctionIOTest : public ::testing::Test {
  virtual void SetUp() {
    host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           host_allocator, &instance));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(instance));
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("test"), host_allocator, host_allocator, &device_allocator));
  }

  virtual void TearDown() {
    iree_hal_allocator_release(device_allocator);
    iree_vm_instance_release(instance);
  }

  Status ParseToVariantList(iree_string_view_t cconv,
                            std::vector<std::string> input_strings,
                            iree_vm_list_t** out_list) {
    std::vector<iree_string_view_t> input_string_views(input_strings.size());
    for (size_t i = 0; i < input_strings.size(); ++i) {
      input_string_views[i].data = input_strings[i].data();
      input_string_views[i].size = input_strings[i].size();
    }
    return iree_tooling_parse_variants(
        cconv,
        iree_string_view_list_t{input_string_views.size(),
                                input_string_views.data()},
        /*device=*/NULL, device_allocator, host_allocator, out_list);
  }

  Status PrintVariantList(iree_vm_list_t* variant_list,
                          std::string* out_string) {
    iree_io_stream_t* stream = NULL;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE,
        /*block_size=*/32 * 1024, host_allocator, &stream));
    iree_status_t status = iree_tooling_print_variants(
        IREE_SV("result"), variant_list, /*max_element_count=*/1024, stream,
        host_allocator);
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
    }
    if (iree_status_is_ok(status)) {
      out_string->resize(iree_io_stream_length(stream));
      status = iree_io_stream_read(stream, out_string->size(),
                                   out_string->data(), NULL);
    }
    iree_io_stream_release(stream);
    return status;
  }

  iree_allocator_t host_allocator;
  iree_vm_instance_t* instance = nullptr;
  iree_hal_allocator_t* device_allocator = nullptr;
};

TEST_F(FunctionIOTest, ParsePrintBuffer) {
  std::string buf_string = "&2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      IREE_SV("r"), std::vector<std::string>{buf_string}, &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer\n") + "(no printer)" + "\n");
}

TEST_F(FunctionIOTest, ParsePrintBufferView) {
  std::string buf_string = "2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      IREE_SV("r"), std::vector<std::string>{buf_string}, &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(FunctionIOTest, ParsePrintScalar) {
  std::string input_string = "42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      IREE_SV("i"), std::vector<std::string>{input_string}, &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result, std::string("result[0]: i32=") + input_string + "\n");
}

TEST_F(FunctionIOTest, ParsePrintRank0BufferView) {
  std::string buf_string = "i32=42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      IREE_SV("r"), std::vector<std::string>{buf_string}, &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result,
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(FunctionIOTest, ParsePrintMultipleBufferViews) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      IREE_SV("rr"), std::vector<std::string>{buf_string1, buf_string2},
      &variant_list));
  std::string result;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &result));
  EXPECT_EQ(result, std::string("result[0]: hal.buffer_view\n") + buf_string1 +
                        "\nresult[1]: hal.buffer_view\n" + buf_string2 + "\n");
}

}  // namespace
}  // namespace iree
