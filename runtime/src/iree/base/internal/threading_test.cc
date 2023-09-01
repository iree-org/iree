// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/threading.h"

#include <chrono>
#include <cstring>
#include <thread>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading_impl.h"  // to test the override list
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;

//==============================================================================
// iree_thread_t
//==============================================================================

TEST(ThreadTest, Lifetime) {
  // Default parameters:
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  // Our thread: do a bit of math and notify the main test thread when done.
  struct entry_data_t {
    iree_atomic_int32_t value;
    iree_notification_t barrier;
  } entry_data;
  iree_atomic_store_int32(&entry_data.value, 123, iree_memory_order_relaxed);
  iree_notification_initialize(&entry_data.barrier);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add_int32(&entry_data->value, 1,
                                iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->barrier, IREE_ALL_WAITERS);
    return 0;
  };

  // Create the thread and immediately begin running it.
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0, iree_thread_id(thread));

  // Wait for the thread to finish.
  iree_notification_await(
      &entry_data.barrier,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load_int32(&entry_data->value,
                                      iree_memory_order_relaxed) == (123 + 1);
      },
      &entry_data, iree_infinite_timeout());

  // By holding on to the thread object and releasing it here after the thread
  // has finished, we ensure that destruction occurs on the main thread,
  // avoiding data races reported by TSan.
  iree_thread_release(thread);
  iree_notification_deinitialize(&entry_data.barrier);
}

TEST(ThreadTest, CreateSuspended) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.create_suspended = true;

  struct entry_data_t {
    iree_atomic_int32_t value;
    iree_notification_t barrier;
  } entry_data;
  iree_atomic_store_int32(&entry_data.value, 123, iree_memory_order_relaxed);
  iree_notification_initialize(&entry_data.barrier);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add_int32(&entry_data->value, 1,
                                iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->barrier, IREE_ALL_WAITERS);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0, iree_thread_id(thread));

  // NOTE: the thread will not be running and we should not expect a change in
  // the value. I can't think of a good way to test this, though, so we'll just
  // wait a moment here and assume that if the thread was able to run it would
  // have during this wait.
  ASSERT_EQ(123, iree_atomic_load_int32(&entry_data.value,
                                        iree_memory_order_seq_cst));
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  ASSERT_EQ(123, iree_atomic_load_int32(&entry_data.value,
                                        iree_memory_order_seq_cst));

  // Resume the thread and wait for it to finish its work.
  iree_thread_resume(thread);
  iree_notification_await(
      &entry_data.barrier,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load_int32(&entry_data->value,
                                      iree_memory_order_relaxed) == (123 + 1);
      },
      &entry_data, iree_infinite_timeout());
  iree_notification_deinitialize(&entry_data.barrier);
  iree_thread_release(thread);
}

// NOTE: testing whether priority took effect is really hard given that on
// certain platforms the priority may not be respected or may be clamped by
// the system. This is here to test the mechanics of the priority override code
// on our side and assumes that if we tell the OS something it respects it.
TEST(ThreadTest, PriorityOverride) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  struct entry_data_t {
    iree_atomic_int32_t value;
  } entry_data;
  iree_atomic_store_int32(&entry_data.value, 0, iree_memory_order_relaxed);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add_int32(&entry_data->value, 1,
                                iree_memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0, iree_thread_id(thread));

  // Push a few overrides.
  // NOTE: some platforms (Apple) may ignore the request and return NULL. Code
  // using overrides needs to be tolerant of this.
  iree_thread_override_t* override0 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGH);
  iree_thread_override_t* override1 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGHEST);
  iree_thread_override_t* override2 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_LOWEST);

  // Wait for the thread to finish.
  while (iree_atomic_load_int32(&entry_data.value, iree_memory_order_acquire) !=
         1) {
    iree_thread_yield();
  }

  // Pop overrides (in opposite order intentionally).
  iree_thread_override_end(override0);
  iree_thread_override_end(override1);
  iree_thread_override_end(override2);

  iree_thread_release(thread);
}

//==============================================================================
// iree_thread_override_list_t
//==============================================================================
// This is an implementation detail but useful to test on its own as it's shared
// across several platform implementations.

TEST(ThreadOverrideListTest, PriorityClass) {
  static iree_thread_t* kThreadSentinel =
      reinterpret_cast<iree_thread_t*>(0x123);
  static iree_thread_priority_class_t current_priority_class =
      IREE_THREAD_PRIORITY_CLASS_NORMAL;
  iree_thread_override_list_t list;
  iree_thread_override_list_initialize(
      +[](iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
        EXPECT_EQ(kThreadSentinel, thread);
        EXPECT_NE(current_priority_class, priority_class);
        current_priority_class = priority_class;
      },
      current_priority_class, iree_allocator_system(), &list);

  // (NORMAL) -> HIGH -> [ignored LOW] -> HIGHEST
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_NORMAL, current_priority_class);
  iree_thread_override_t* override0 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_HIGH);
  EXPECT_NE(nullptr, override0);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGH, current_priority_class);
  iree_thread_override_t* override1 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_LOW);
  EXPECT_NE(nullptr, override1);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGH, current_priority_class);
  iree_thread_override_t* override2 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_HIGHEST);
  EXPECT_NE(nullptr, override2);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);

  // Out of order to ensure highest bit sticks:
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override1);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override0);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override2);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_NORMAL, current_priority_class);

  iree_thread_override_list_deinitialize(&list);
}

}  // namespace
