// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/atomics.h"

#include <cstddef>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

// NOTE: these tests are just to ensure we correctly compile the macros across
// our supported toolchains: they don't verify that the memory semantics are
// correct (as that would be difficult and is really the toolchain's job).

TEST(AtomicPtr, LoadStore) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0, iree_atomic_load(&value, iree_memory_order_seq_cst));
  iree_atomic_store(&value, ptr_1, iree_memory_order_seq_cst);
  EXPECT_EQ(ptr_1, iree_atomic_load(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, AddSub) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0,
            iree_atomic_fetch_add(&value, ptr_1, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1,
            iree_atomic_fetch_add(&value, ptr_1, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_2,
            iree_atomic_fetch_sub(&value, ptr_1, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1,
            iree_atomic_fetch_sub(&value, ptr_1, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, iree_atomic_load(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, Exchange) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  EXPECT_EQ(ptr_0,
            iree_atomic_exchange(&value, ptr_1, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1,
            iree_atomic_exchange(&value, ptr_2, iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_2, iree_atomic_load(&value, iree_memory_order_seq_cst));
}

TEST(AtomicPtr, CompareExchange) {
  intptr_t ptr_0 = 0x0;
  intptr_t ptr_1 = 0x1;
  intptr_t ptr_2 = 0x2;
  iree_atomic_intptr_t value = IREE_ATOMIC_VAR_INIT(ptr_0);
  intptr_t ptr_expected = 0;

  // OK: value == ptr_0, CAS(ptr_0 -> ptr_1)
  iree_atomic_store(&value, ptr_0, iree_memory_order_seq_cst);
  ptr_expected = ptr_0;
  EXPECT_TRUE(iree_atomic_compare_exchange_strong(&value, &ptr_expected, ptr_1,
                                                  iree_memory_order_seq_cst,
                                                  iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, ptr_expected);
  EXPECT_EQ(ptr_1, iree_atomic_load(&value, iree_memory_order_seq_cst));

  // OK: value == ptr_1, CAS(ptr_1 -> ptr_2)
  iree_atomic_store(&value, ptr_1, iree_memory_order_seq_cst);
  ptr_expected = ptr_1;
  EXPECT_TRUE(iree_atomic_compare_exchange_strong(&value, &ptr_expected, ptr_2,
                                                  iree_memory_order_seq_cst,
                                                  iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_1, ptr_expected);
  EXPECT_EQ(ptr_2, iree_atomic_load(&value, iree_memory_order_seq_cst));

  // FAIL: value == ptr_0, CAS(ptr_1 -> ptr_2)
  iree_atomic_store(&value, ptr_0, iree_memory_order_seq_cst);
  ptr_expected = ptr_1;
  EXPECT_FALSE(iree_atomic_compare_exchange_strong(&value, &ptr_expected, ptr_2,
                                                   iree_memory_order_seq_cst,
                                                   iree_memory_order_seq_cst));
  EXPECT_EQ(ptr_0, ptr_expected);
  EXPECT_EQ(ptr_0, iree_atomic_load(&value, iree_memory_order_seq_cst));
}

TEST(AtomicRefCount, IncDec) {
  iree_atomic_ref_count_t count;
  iree_atomic_ref_count_init(&count);
  iree_atomic_ref_count_inc(&count);
  iree_atomic_ref_count_inc(&count);
  EXPECT_EQ(3, iree_atomic_ref_count_dec(&count));
  EXPECT_EQ(2, iree_atomic_ref_count_dec(&count));
  EXPECT_EQ(1, iree_atomic_ref_count_dec(&count));
}

}  // namespace
