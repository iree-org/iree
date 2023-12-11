// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_parser.h"

#include "iree/io/formats/irpa/testdata/irpa_files.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static iree_io_file_handle_t* OpenTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_io_irpa_files_create();
  for (size_t i = 0; i < iree_io_irpa_files_size(); ++i) {
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

TEST(IrpaFormatTest, Empty) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("empty.irpa");
  IREE_ASSERT_OK(iree_io_parse_irpa_index(file_handle, index));
  EXPECT_EQ(0, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, SingleParameters) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("single.irpa");
  IREE_ASSERT_OK(iree_io_parse_irpa_index(file_handle, index));
  EXPECT_EQ(1, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 192);
  EXPECT_EQ(entry0->length, 16);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, MultipleParameters) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("multiple.irpa");
  IREE_ASSERT_OK(iree_io_parse_irpa_index(file_handle, index));
  EXPECT_EQ(2, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 320);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry1->storage.file.offset, 384);
  EXPECT_EQ(entry1->length, 5);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, MixedDataAndSplats) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("mixed.irpa");
  IREE_ASSERT_OK(iree_io_parse_irpa_index(file_handle, index));
  EXPECT_EQ(4, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 512);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry1->storage.file.offset, 576);
  EXPECT_EQ(entry1->length, 5);

  const iree_io_parameter_index_entry_t* entry2 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key2"), &entry2));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key2"), entry2->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry2->metadata));
  EXPECT_EQ(entry2->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT);
  EXPECT_EQ(entry2->storage.splat.pattern_length, 1);
  const int8_t entry2_pattern = 102;
  EXPECT_EQ(0, memcmp(&entry2_pattern, entry2->storage.splat.pattern,
                      sizeof(entry2_pattern)));
  EXPECT_EQ(entry2->length, 1);

  const iree_io_parameter_index_entry_t* entry3 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key3"), &entry3));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key3"), entry3->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry3->metadata));
  EXPECT_EQ(entry3->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT);
  EXPECT_EQ(entry3->storage.splat.pattern_length, 8);
  const int64_t entry3_pattern = 9223372036854775807ll;
  EXPECT_EQ(0, memcmp(&entry3_pattern, entry3->storage.splat.pattern,
                      sizeof(entry3_pattern)));
  EXPECT_EQ(entry3->length, 33554432);

  iree_io_parameter_index_release(index);
}

}  // namespace
}  // namespace iree
