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

#include "iree/base/internal/atomics.h"

#include "iree/testing/gtest.h"

namespace {

// NOTE: these tests are just to ensure we correctly compile the macros across
// our supported toolchains: they don't verify that the memory semantics are
// correct (as that would be difficult and is really the toolchain's job).

TEST(AtomicPtr, LoadStore) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));
  iree_atomic_store_intptr(&value, ptr_1, iree_memory_order_seq_cst);
  EXPECT_EQ(ptr_1, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, AddSub) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0, iree_atomic_fetch_add_intptr(&value, ptr_1,
                                                iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1, iree_atomic_fetch_add_intptr(&value, ptr_1,
                                                iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_2, iree_atomic_fetch_sub_intptr(&value, ptr_1,
                                                iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1, iree_atomic_fetch_sub_intptr(&value, ptr_1,
                                                iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, Exchange) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0, iree_atomic_exchange_intptr(&value, ptr_1,
                                               iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1, iree_atomic_exchange_intptr(&value, ptr_2,
                                               iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_2, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, CompareExchange) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  intptr_t ptr_expected = NULL;

  // OK: value == ptr_0, CAS(ptr_0 -> ptr_1)
  iree_atomic_store_intptr(&value, ptr_0, iree_memory_order_seq_cst);
  ptr_expected = ptr_0;
  EXPECT_TRUE(iree_atomic_compare_exchange_strong_intptr(
      &value, &ptr_expected, ptr_1, iree_memory_order_seq_cst,
      iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, ptr_expected);
  EXPECT_EQ(ptr_1, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));

  // OK: value == ptr_1, CAS(ptr_1 -> ptr_2)
  iree_atomic_store_intptr(&value, ptr_1, iree_memory_order_seq_cst);
  ptr_expected = ptr_1;
  EXPECT_TRUE(iree_atomic_compare_exchange_strong_intptr(
      &value, &ptr_expected, ptr_2, iree_memory_order_seq_cst,
      iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1, ptr_expected);
  EXPECT_EQ(ptr_2, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));

  // FAIL: value == ptr_0, CAS(ptr_1 -> ptr_2)
  iree_atomic_store_intptr(&value, ptr_0, iree_memory_order_seq_cst);
  ptr_expected = ptr_1;
  EXPECT_FALSE(iree_atomic_compare_exchange_strong_intptr(
      &value, &ptr_expected, ptr_2, iree_memory_order_seq_cst,
      iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, ptr_expected);
  EXPECT_EQ(ptr_0, iree_atomic_load_intptr(&value, iree_memory_order_seq_cst));
}

TEST(AtomicRefCount, IncDec) {
  iree_atomic_ref_count_t count;
  iree_atomic_ref_count_init(&count);
  EXPECT_EQ(1, iree_atomic_ref_count_inc(&count));
  EXPECT_EQ(2, iree_atomic_ref_count_inc(&count));
  EXPECT_EQ(3, iree_atomic_ref_count_dec(&count));
  EXPECT_EQ(2, iree_atomic_ref_count_dec(&count));
  EXPECT_EQ(1, iree_atomic_ref_count_dec(&count));
}

}  // namespace
