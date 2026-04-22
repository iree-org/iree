// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/common.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ProfileCommonTest, NamesProfileStatusCodes) {
  EXPECT_STREQ("CANCELLED",
               iree_profile_status_code_name(IREE_STATUS_CANCELLED));
  EXPECT_STREQ("UNKNOWN_STATUS", iree_profile_status_code_name(UINT32_MAX));
}

typedef struct IndexLookup {
  // Candidate key array addressed by the index value.
  const uint64_t* keys;
  // Lookup key being searched for.
  uint64_t key;
} IndexLookup;

static bool IndexLookupMatches(const void* user_data, iree_host_size_t value) {
  const IndexLookup* lookup = static_cast<const IndexLookup*>(user_data);
  return lookup->keys[value] == lookup->key;
}

TEST(ProfileCommonTest, IndexFindsInsertedRows) {
  uint64_t keys[] = {11, 22, 33};
  iree_profile_index_t index = {0};
  IREE_EXPECT_OK(iree_profile_index_insert(
      &index, iree_allocator_system(), iree_profile_index_mix_u64(keys[0]), 0));
  IREE_EXPECT_OK(iree_profile_index_insert(
      &index, iree_allocator_system(), iree_profile_index_mix_u64(keys[1]), 1));
  IREE_EXPECT_OK(iree_profile_index_insert(
      &index, iree_allocator_system(), iree_profile_index_mix_u64(keys[2]), 2));

  IndexLookup lookup = {keys, 22};
  iree_host_size_t value = IREE_HOST_SIZE_MAX;
  EXPECT_TRUE(iree_profile_index_find(&index, iree_profile_index_mix_u64(22),
                                      IndexLookupMatches, &lookup, &value));
  EXPECT_EQ(1u, value);

  lookup.key = 44;
  EXPECT_FALSE(iree_profile_index_find(&index, iree_profile_index_mix_u64(44),
                                       IndexLookupMatches, &lookup, &value));
  iree_profile_index_deinitialize(&index, iree_allocator_system());
}

TEST(ProfileCommonTest, IndexResolvesHashCollisionsWithEquality) {
  uint64_t keys[] = {7, 9};
  iree_profile_index_t index = {0};
  IREE_EXPECT_OK(
      iree_profile_index_insert(&index, iree_allocator_system(), 1, 0));
  IREE_EXPECT_OK(
      iree_profile_index_insert(&index, iree_allocator_system(), 1, 1));

  IndexLookup lookup = {keys, 9};
  iree_host_size_t value = IREE_HOST_SIZE_MAX;
  EXPECT_TRUE(
      iree_profile_index_find(&index, 1, IndexLookupMatches, &lookup, &value));
  EXPECT_EQ(1u, value);

  lookup.key = 7;
  EXPECT_TRUE(
      iree_profile_index_find(&index, 1, IndexLookupMatches, &lookup, &value));
  EXPECT_EQ(0u, value);
  iree_profile_index_deinitialize(&index, iree_allocator_system());
}

}  // namespace
