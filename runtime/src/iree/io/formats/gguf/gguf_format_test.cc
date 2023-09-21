// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/gguf/gguf_format.h"

#include "iree/base/internal/file_io.h"
#include "iree/io/formats/gguf/testdata/gguf_files.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static iree_io_file_handle_t* OpenTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_io_gguf_files_create();
  for (size_t i = 0; i < iree_io_gguf_files_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      iree_io_file_handle_t* file_handle = NULL;
      IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ,
          iree_make_byte_span((void*)file_toc[i].data, file_toc[i].size),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          &file_handle));
      return file_handle;
    }
  }
  IREE_CHECK_OK(iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "test file `%s` not found embedded into test binary", name));
  return NULL;
}

TEST(GgufFormatTest, Empty) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("empty.gguf");
  IREE_ASSERT_OK(iree_io_parse_gguf_index(file_handle, index));
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, SingleTensors) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("single.gguf");
  IREE_ASSERT_OK(iree_io_parse_gguf_index(file_handle, index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->offset, 384);
  EXPECT_EQ(entry0->length, 16);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, MultipleTensors) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("multiple.gguf");
  IREE_ASSERT_OK(iree_io_parse_gguf_index(file_handle, index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->offset, 448);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->offset, 512);
  EXPECT_EQ(entry1->length, 8);

  const iree_io_parameter_index_entry_t* entry2 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor2"), &entry2));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor2"), entry2->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry2->metadata));
  EXPECT_EQ(entry2->offset, 576);
  EXPECT_EQ(entry2->length, 48);

  iree_io_parameter_index_release(index);
}

}  // namespace
}  // namespace iree
