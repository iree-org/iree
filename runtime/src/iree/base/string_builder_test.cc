// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

struct StringBuilder {
  static StringBuilder MakeSystem() {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    return StringBuilder(builder);
  }

  static StringBuilder MakeEmpty() {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_null(), &builder);
    return StringBuilder(builder);
  }

  explicit StringBuilder(iree_string_builder_t builder)
      : builder(std::move(builder)) {}

  ~StringBuilder() { iree_string_builder_deinitialize(&builder); }

  operator iree_string_builder_t*() { return &builder; }

  std::string ToString() const {
    return std::string(builder.buffer, builder.size);
  }

  iree_string_builder_t builder;

 protected:
  StringBuilder() = default;
};

template <size_t Capacity>
struct InlineStringBuilder : public StringBuilder {
  InlineStringBuilder() {
    iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                                &builder);
  }
  char storage[Capacity] = {0};
};

struct StringPairBuilder {
  static StringPairBuilder MakeSystem() {
    iree_string_pair_builder_t builder;
    iree_string_pair_builder_initialize(iree_allocator_system(), &builder);
    return StringPairBuilder(builder);
  }

  explicit StringPairBuilder(iree_string_pair_builder_t builder)
      : builder(std::move(builder)) {}

  ~StringPairBuilder() { iree_string_pair_builder_deinitialize(&builder); }

  operator iree_string_pair_builder_t*() { return &builder; }

  iree_string_pair_builder_t builder;

  std::string PairFirstToString(iree_host_size_t i) {
    iree_string_pair_t* pair = iree_string_pair_builder_pairs(&builder) + i;
    return std::string(pair->first.data, pair->first.size);
  }

  std::string PairSecondToString(iree_host_size_t i) {
    iree_string_pair_t* pair = iree_string_pair_builder_pairs(&builder) + i;
    return std::string(pair->second.data, pair->second.size);
  }

 protected:
  StringPairBuilder() = default;
};

TEST(StringBuilderTest, QueryEmpty) {
  auto builder = StringBuilder::MakeEmpty();
  EXPECT_EQ(iree_string_builder_buffer(builder),
            static_cast<const char*>(NULL));
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  EXPECT_EQ(iree_string_builder_capacity(builder), 0);
  EXPECT_TRUE(iree_string_view_is_empty(iree_string_builder_view(builder)));
  EXPECT_EQ(iree_string_builder_take_storage(builder),
            static_cast<char*>(NULL));
}

TEST(StringBuilderTest, QueryAppendString) {
  auto builder = StringBuilder::MakeEmpty();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, ""));
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "a"));
  EXPECT_EQ(iree_string_builder_size(builder), 1);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "abc"));
  EXPECT_EQ(iree_string_builder_size(builder), 1 + 3);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, ""));
  EXPECT_EQ(iree_string_builder_size(builder), 1 + 3);

  char kLongString[1024];
  memset(kLongString, 'x', IREE_ARRAYSIZE(kLongString));
  IREE_EXPECT_OK(iree_string_builder_append_string(
      builder,
      iree_make_string_view(kLongString, IREE_ARRAYSIZE(kLongString))));
  EXPECT_EQ(iree_string_builder_size(builder),
            1 + 3 + IREE_ARRAYSIZE(kLongString));
}

TEST(StringBuilderTest, QueryFormat) {
  auto builder = StringBuilder::MakeEmpty();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, ""));
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "abc"));
  EXPECT_EQ(iree_string_builder_size(builder), 3);
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "a%cc", 'b'));
  EXPECT_EQ(iree_string_builder_size(builder), 6);
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "%*c", 1024, 'x'));
  EXPECT_EQ(iree_string_builder_size(builder), 6 + 1024);
}

TEST(StringBuilderTest, Empty) {
  auto builder = StringBuilder::MakeSystem();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  EXPECT_GE(iree_string_builder_capacity(builder), 0);
  EXPECT_TRUE(iree_string_view_is_empty(iree_string_builder_view(builder)));
  EXPECT_EQ(iree_string_builder_take_storage(builder),
            static_cast<char*>(NULL));
}

TEST(StringBuilderTest, AppendString) {
  auto builder = StringBuilder::MakeSystem();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, ""));
  EXPECT_EQ(builder.ToString(), "");
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "a"));
  EXPECT_EQ(builder.ToString(), "a");
  EXPECT_EQ(strlen(builder.builder.buffer), 1);  // NUL check
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "abc"));
  EXPECT_EQ(builder.ToString(), "aabc");
  EXPECT_EQ(strlen(builder.builder.buffer), 1 + 3);  // NUL check
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, ""));
  EXPECT_EQ(builder.ToString(), "aabc");
  EXPECT_EQ(iree_string_builder_size(builder), 1 + 3);
  EXPECT_EQ(strlen(builder.builder.buffer), 1 + 3);  // NUL check

  char kLongString[1024];
  memset(kLongString, 'x', IREE_ARRAYSIZE(kLongString));
  IREE_EXPECT_OK(iree_string_builder_append_string(
      builder,
      iree_make_string_view(kLongString, IREE_ARRAYSIZE(kLongString))));
  EXPECT_EQ(iree_string_builder_size(builder),
            1 + 3 + IREE_ARRAYSIZE(kLongString));
  EXPECT_EQ(strlen(builder.builder.buffer),
            1 + 3 + IREE_ARRAYSIZE(kLongString));  // NUL check
  EXPECT_EQ(builder.ToString(),
            std::string("aabc") +
                std::string(kLongString, IREE_ARRAYSIZE(kLongString)));
}

TEST(StringBuilderTest, TakeStorage) {
  auto builder = StringBuilder::MakeSystem();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "a"));
  EXPECT_EQ(builder.ToString(), "a");
  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "abc"));
  EXPECT_EQ(builder.ToString(), "aabc");
  EXPECT_EQ(iree_string_builder_size(builder), 1 + 3);
  EXPECT_EQ(strlen(builder.builder.buffer),
            1 + 3);  // NUL check

  char* storage = iree_string_builder_take_storage(builder);
  EXPECT_EQ(iree_string_builder_buffer(builder),
            static_cast<const char*>(NULL));
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  EXPECT_EQ(iree_string_builder_capacity(builder), 0);
  EXPECT_NE(storage, static_cast<char*>(NULL));
  EXPECT_STREQ(storage, "aabc");
  EXPECT_EQ(builder.builder.buffer, static_cast<char*>(NULL));
  iree_allocator_free(builder.builder.allocator, storage);
}

TEST(StringBuilderTest, Format) {
  auto builder = StringBuilder::MakeSystem();
  EXPECT_EQ(builder.ToString(), "");
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, ""));
  EXPECT_EQ(builder.ToString(), "");
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "abc"));
  EXPECT_EQ(builder.ToString(), "abc");
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "a%cc", 'b'));
  EXPECT_EQ(builder.ToString(), "abcabc");
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "%*c", 1024, 'x'));
  EXPECT_EQ(iree_string_builder_size(builder), 6 + 1024);
  EXPECT_EQ(strlen(builder.builder.buffer), 6 + 1024);  // NUL check
  EXPECT_EQ(builder.ToString(),
            std::string("abcabc") + std::string(1023, ' ') + std::string("x"));
}

TEST(StringBuilderTest, InlineStorage) {
  InlineStringBuilder<8> builder;
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  EXPECT_GE(iree_string_builder_capacity(builder), 8);
  EXPECT_TRUE(iree_string_view_is_empty(iree_string_builder_view(builder)));

  // Should be able to reserve up to capacity.
  IREE_EXPECT_OK(iree_string_builder_reserve(builder, 4));
  IREE_EXPECT_OK(iree_string_builder_reserve(builder, 8));

  // Should fail to reserve more than storage size.
  EXPECT_THAT(Status(iree_string_builder_reserve(builder, 9)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(StringBuilderTest, SizeCalculation) {
  auto builder = StringBuilder::MakeEmpty();
  EXPECT_EQ(iree_string_builder_size(builder), 0);
  EXPECT_GE(iree_string_builder_capacity(builder), 0);
  EXPECT_TRUE(iree_string_view_is_empty(iree_string_builder_view(builder)));

  IREE_EXPECT_OK(iree_string_builder_append_cstring(builder, "abc"));
  EXPECT_EQ(iree_string_builder_size(builder), 3);
  EXPECT_GE(iree_string_builder_capacity(builder), 0);

  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "def"));
  EXPECT_EQ(iree_string_builder_size(builder), 6);
  EXPECT_GE(iree_string_builder_capacity(builder), 0);

  char* head = NULL;
  IREE_EXPECT_OK(iree_string_builder_append_inline(builder, 3, &head));
  EXPECT_TRUE(head == NULL);
  EXPECT_EQ(iree_string_builder_size(builder), 9);
  EXPECT_GE(iree_string_builder_capacity(builder), 0);

  // Reservation should fail because there's no allocator.
  EXPECT_THAT(Status(iree_string_builder_reserve(builder, 4)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(StringBuilderTest, ReserveForAppend) {
  auto builder = StringBuilder::MakeSystem();
  IREE_EXPECT_OK(iree_string_builder_append_format(builder, "def"));
  char* append_buffer;
  iree_host_size_t append_capacity;
  // 4096 is big enough that it will exceed the initial allocation.
  IREE_EXPECT_OK(iree_string_builder_reserve_for_append(
      builder, 4096, &append_buffer, &append_capacity));
  ASSERT_GE(append_capacity, 4096);
  memset(append_buffer, 0xff, append_capacity);
  memcpy(append_buffer, ":hello", 6);
  iree_string_builder_commit_append(builder, 6);
  EXPECT_EQ(builder.ToString(), std::string("def:hello"));
  EXPECT_EQ(builder.builder.buffer[builder.builder.size], 0);
}

TEST(StringPairBuilderTest, ReallocAndSize) {
  auto builder = StringPairBuilder::MakeSystem();
  ASSERT_EQ(iree_string_pair_builder_size(builder), 0);
  IREE_EXPECT_OK(iree_string_pair_builder_add(
      builder, iree_make_cstring_pair("name", "value")));
  auto orig_capacity = builder.builder.pairs_capacity;
  ASSERT_GE(orig_capacity, 1);
  for (iree_host_size_t i = 0; i < orig_capacity + 1; ++i) {
    IREE_EXPECT_OK(iree_string_pair_builder_add(
        builder, iree_make_cstring_pair("name", "value")));
  }
  ASSERT_GT(builder.builder.pairs_capacity, orig_capacity);

  iree_host_size_t expected_size = orig_capacity + 2;
  ASSERT_EQ(iree_string_pair_builder_size(builder), expected_size);
  for (iree_host_size_t i = 0; i < expected_size; ++i) {
    EXPECT_EQ(builder.PairFirstToString(i), "name");
    EXPECT_EQ(builder.PairSecondToString(i), "value");
  }
}

TEST(StringPairBuilderTest, AddInt32) {
  auto builder = StringPairBuilder::MakeSystem();
  IREE_EXPECT_OK(iree_string_pair_builder_add_int32(
      builder, iree_make_cstring_view("key"), 1));
  IREE_EXPECT_OK(iree_string_pair_builder_add_int32(
      builder, iree_make_cstring_view("key"), 0));
  EXPECT_EQ(builder.PairFirstToString(0), "key");
  EXPECT_EQ(builder.PairSecondToString(0), "1");
  EXPECT_EQ(builder.PairFirstToString(1), "key");
  EXPECT_EQ(builder.PairSecondToString(1), "0");
}

TEST(StringPairBuilderTest, EmplaceString) {
  auto builder = StringPairBuilder::MakeSystem();
  std::vector<iree_string_view_t> all_emplaced;
  // Just ensure a couple of reallocs.
  for (int i = 0; i < 128; ++i) {
    iree_string_view_t emplaced = iree_make_cstring_view("foobar");
    iree_string_view_t original = emplaced;
    IREE_EXPECT_OK(iree_string_pair_builder_emplace_string(builder, &emplaced));
    EXPECT_NE(emplaced.data, original.data);
    all_emplaced.push_back(emplaced);
  }

  for (iree_string_view_t emplaced : all_emplaced) {
    EXPECT_EQ(
        iree_string_view_compare(emplaced, iree_make_cstring_view("foobar")),
        0);
  }
}

}  // namespace
