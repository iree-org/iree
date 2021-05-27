// Copyright 2020 Google LLC
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

#include "iree/base/internal/atomic_slist.h"

#include "iree/testing/gtest.h"

namespace {

struct dummy_entry_t {
  // NOTE: we purposefully offset the entry pointer
  size_t value = 0;
  iree_atomic_slist_intrusive_ptr_t slist_next = NULL;
};
IREE_TYPED_ATOMIC_SLIST_WRAPPER(dummy, dummy_entry_t,
                                offsetof(dummy_entry_t, slist_next));

std::vector<dummy_entry_t> MakeDummySListItems(size_t base_index,
                                               size_t count) {
  std::vector<dummy_entry_t> items(count);
  for (size_t i = 0; i < count; ++i) {
    items[i].value = base_index + i;
  }
  return items;
}

TEST(AtomicSList, Lifetime) {
  iree_atomic_slist_t list;  // NOTE: intentionally uninitialized.
  iree_atomic_slist_initialize(&list);
  iree_atomic_slist_deinitialize(&list);
}

TEST(AtomicSList, BasicUsage) {
  dummy_slist_t list;
  dummy_slist_initialize(&list);

  // List starts empty.
  EXPECT_EQ(NULL, dummy_slist_pop(&list));

  // Push some items into the list (LIFO order).
  // New contents: 5 4 3 2 1 0
  auto item_storage = MakeDummySListItems(0, 6);
  for (size_t i = 0; i < item_storage.size(); ++i) {
    dummy_slist_push(&list, &item_storage[i]);
  }

  // Now pop them out - they should be in reverse order.
  // New contents: e
  for (size_t i = 0; i < item_storage.size(); ++i) {
    dummy_entry_t* p = dummy_slist_pop(&list);
    ASSERT_TRUE(p);
    EXPECT_EQ(item_storage.size() - i - 1, p->value);
  }

  // List ends empty.
  EXPECT_EQ(NULL, dummy_slist_pop(&list));

  dummy_slist_deinitialize(&list);
}

TEST(AtomicSList, Concat) {
  dummy_slist_t list;
  dummy_slist_initialize(&list);

  // Push some initial items into the list (LIFO order).
  // New contents: 1 0
  auto initial_item_storage = MakeDummySListItems(0, 2);
  for (size_t i = 0; i < initial_item_storage.size(); ++i) {
    dummy_slist_push(&list, &initial_item_storage[i]);
  }

  // Stitch items together modeling what a user may do when building the list
  // themselves.
  // Items: 2 3 4
  auto span_item_storage = MakeDummySListItems(2, 3);
  for (size_t i = 0; i < span_item_storage.size() - 1; ++i) {
    dummy_slist_set_next(&span_item_storage[i], &span_item_storage[i + 1]);
  }

  // Push all of the items to the list at once.
  // New contents: 2 3 4 1 0
  dummy_slist_concat(&list, &span_item_storage.front(),
                     &span_item_storage.back());

  // Pop the span items and verify they are in the correct order: we effectively
  // pushed them such that popping is FIFO (2->4).
  // New contents: 1 0
  for (size_t i = 0; i < span_item_storage.size(); ++i) {
    dummy_entry_t* p = dummy_slist_pop(&list);
    ASSERT_TRUE(p);
    EXPECT_EQ(/*base_index=*/2 + i, p->value);
  }

  // Pop the initial items and ensure they survived.
  // New contents: e
  for (size_t i = 0; i < initial_item_storage.size(); ++i) {
    dummy_entry_t* p = dummy_slist_pop(&list);
    ASSERT_TRUE(p);
    EXPECT_EQ(initial_item_storage.size() - i - 1, p->value);
  }

  dummy_slist_deinitialize(&list);
}

TEST(AtomicSList, FlushLIFO) {
  dummy_slist_t list;
  dummy_slist_initialize(&list);

  // Flushing when empty is ok.
  dummy_entry_t* head = NULL;
  dummy_entry_t* tail = NULL;
  EXPECT_FALSE(dummy_slist_flush(
      &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, &tail));

  // Push items into the list (LIFO order).
  // New contents: 3 2 1 0
  auto item_storage = MakeDummySListItems(0, 4);
  for (size_t i = 0; i < item_storage.size(); ++i) {
    dummy_slist_push(&list, &item_storage[i]);
  }

  // Flush in LIFO order and verify empty.
  // New contents: e
  EXPECT_TRUE(dummy_slist_flush(
      &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, &tail));
  EXPECT_EQ(NULL, dummy_slist_pop(&list));

  // Verify LIFO order and list pointer walking.
  // Note that head and tail are reverse of item storage!
  EXPECT_EQ(&item_storage.back(), head);
  EXPECT_EQ(&item_storage.front(), tail);
  dummy_entry_t* p = head;
  for (size_t i = 0; i < item_storage.size(); ++i) {
    ASSERT_TRUE(p);
    EXPECT_EQ(item_storage.size() - i - 1, p->value);
    p = dummy_slist_get_next(p);
  }
  EXPECT_EQ(NULL, p);

  dummy_slist_deinitialize(&list);
}

TEST(AtomicSList, FlushFIFO) {
  dummy_slist_t list;
  dummy_slist_initialize(&list);

  // Flushing when empty is ok.
  dummy_entry_t* head = NULL;
  dummy_entry_t* tail = NULL;
  EXPECT_FALSE(dummy_slist_flush(
      &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO, &head, &tail));

  // Push items into the list (LIFO order).
  // New contents: 3 2 1 0
  auto item_storage = MakeDummySListItems(0, 4);
  for (size_t i = 0; i < item_storage.size(); ++i) {
    dummy_slist_push(&list, &item_storage[i]);
  }

  // Flush in FIFO order and verify empty.
  // New contents: e
  EXPECT_TRUE(dummy_slist_flush(
      &list, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO, &head, &tail));
  EXPECT_EQ(NULL, dummy_slist_pop(&list));

  // Verify FIFO order and list pointer walking.
  EXPECT_EQ(&item_storage.front(), head);
  EXPECT_EQ(&item_storage.back(), tail);
  dummy_entry_t* p = head;
  for (size_t i = 0; i < item_storage.size(); ++i) {
    ASSERT_TRUE(p);
    EXPECT_EQ(i, p->value);
    p = dummy_slist_get_next(p);
  }
  EXPECT_EQ(NULL, p);

  dummy_slist_deinitialize(&list);
}

}  // namespace
