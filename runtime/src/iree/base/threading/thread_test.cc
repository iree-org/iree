// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/thread.h"

#include <chrono>
#include <cstring>
#include <thread>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread_impl.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//==============================================================================
// iree_thread_affinity_t
//==============================================================================

TEST(ThreadAffinityTest, SetAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0xFF, sizeof(affinity));  // Dirty memory.
  iree_thread_affinity_set_any(&affinity);
  EXPECT_TRUE(iree_thread_affinity_is_unspecified(affinity));
  EXPECT_FALSE(affinity.group_any);
  EXPECT_FALSE(affinity.id_assigned);
}

TEST(ThreadAffinityTest, SetGroupAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0xFF, sizeof(affinity));  // Dirty memory.
  iree_thread_affinity_set_group_any(5, &affinity);
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
  EXPECT_TRUE(affinity.group_any);
  EXPECT_EQ(5u, affinity.group);
  EXPECT_FALSE(affinity.id_assigned);
}

TEST(ThreadAffinityTest, IsUnspecifiedFalseWhenGroupAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0, sizeof(affinity));
  affinity.group_any = 1;
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
}

TEST(ThreadAffinityTest, IsUnspecifiedFalseWhenIdAssigned) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0, sizeof(affinity));
  affinity.id_assigned = 1;
  affinity.id = 3;
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
}

//==============================================================================
// iree_thread_t
//==============================================================================

TEST(ThreadTest, Lifetime) {
  // Default parameters.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  // Our thread: do a bit of math and notify the main test thread when done.
  struct entry_data_t {
    iree_atomic_int32_t value;
    iree_notification_t barrier;
  } entry_data;
  iree_atomic_store(&entry_data.value, 123, iree_memory_order_relaxed);
  iree_notification_initialize(&entry_data.barrier);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->barrier, IREE_ALL_WAITERS);
    return 0;
  };

  // Create the thread and immediately begin running it.
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Wait for the thread to finish.
  iree_notification_await(
      &entry_data.barrier,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load(&entry_data->value,
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
    std::atomic<bool> gate_open;
    iree_notification_t gate;
    iree_notification_t done;
  } entry_data;
  iree_atomic_store(&entry_data.value, 123, iree_memory_order_relaxed);
  entry_data.gate_open.store(false, std::memory_order_relaxed);
  iree_notification_initialize(&entry_data.gate);
  iree_notification_initialize(&entry_data.done);

  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    // Wait for gate to open before doing work.
    iree_notification_await(
        &entry_data->gate,
        +[](void* arg) {
          return static_cast<std::atomic<bool>*>(arg)->load(
              std::memory_order_acquire);
        },
        &entry_data->gate_open, iree_infinite_timeout());
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->done, IREE_ALL_WAITERS);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Value should be unchanged (thread created but suspended).
  ASSERT_EQ(123,
            iree_atomic_load(&entry_data.value, iree_memory_order_seq_cst));

  // Open the gate. If the thread were running (not suspended), it would now
  // pass through the gate and modify the value. Since it's suspended, the
  // value should remain unchanged.
  entry_data.gate_open.store(true, std::memory_order_release);
  iree_notification_post(&entry_data.gate, IREE_ALL_WAITERS);

  // Value should still be unchanged (thread is suspended, can't run).
  ASSERT_EQ(123,
            iree_atomic_load(&entry_data.value, iree_memory_order_seq_cst));

  // Resume the thread and wait for it to finish its work.
  iree_thread_resume(thread);
  iree_notification_await(
      &entry_data.done,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load(&entry_data->value,
                                iree_memory_order_relaxed) == (123 + 1);
      },
      &entry_data, iree_infinite_timeout());

  iree_thread_release(thread);
  iree_notification_deinitialize(&entry_data.done);
  iree_notification_deinitialize(&entry_data.gate);
}

TEST(ThreadTest, RetainRelease) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  std::atomic<bool> completed{false};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* completed = reinterpret_cast<std::atomic<bool>*>(entry_arg);
    completed->store(true, std::memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &completed, params,
                                    iree_allocator_system(), &thread));

  // Retain adds a reference.
  iree_thread_retain(thread);

  // Wait for thread to complete.
  while (!completed.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // First release does not destroy the thread.
  iree_thread_release(thread);

  // Verify we can still get the thread ID (object is alive).
  EXPECT_NE(0u, iree_thread_id(thread));

  // Second release destroys the thread.
  iree_thread_release(thread);
}

TEST(ThreadTest, ReleaseWaitsForCompletion) {
  // Tests that iree_thread_release properly waits for thread completion.
  // Note: iree_thread_join + iree_thread_release cannot both be called as
  // release internally joins when destroying the thread.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  std::atomic<int32_t> value{0};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* value = reinterpret_cast<std::atomic<int32_t>*>(entry_arg);
    // Do some work.
    for (int i = 0; i < 100; ++i) {
      value->fetch_add(1, std::memory_order_relaxed);
      iree_thread_yield();
    }
    return 42;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &value, params,
                                    iree_allocator_system(), &thread));

  // Release internally joins/waits for the thread to complete.
  iree_thread_release(thread);

  // Thread has completed all work (release waited for it).
  EXPECT_EQ(100, value.load(std::memory_order_acquire));
}

TEST(ThreadTest, NamedThread) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("TestThread");

  std::atomic<bool> completed{false};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* completed = reinterpret_cast<std::atomic<bool>*>(entry_arg);
    completed->store(true, std::memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &completed, params,
                                    iree_allocator_system(), &thread));

  // Wait for completion.
  while (!completed.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  iree_thread_release(thread);
}

TEST(ThreadTest, RequestAffinity) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  std::atomic<bool> completed{false};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* completed = reinterpret_cast<std::atomic<bool>*>(entry_arg);
    // Spin for a bit to give time for affinity to take effect (maybe).
    for (int i = 0; i < 1000; ++i) {
      iree_thread_yield();
    }
    completed->store(true, std::memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &completed, params,
                                    iree_allocator_system(), &thread));

  // Request affinity to group 0 (smoke test - may be ignored by OS).
  iree_thread_affinity_t affinity;
  iree_thread_affinity_set_group_any(0, &affinity);
  iree_thread_request_affinity(thread, affinity);

  // Wait for completion.
  while (!completed.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  iree_thread_release(thread);
}

// Testing whether priority took effect is hard given that on certain platforms
// the priority may not be respected or may be clamped by the system. This test
// exercises the mechanics of the priority override code on our side.
TEST(ThreadTest, PriorityOverride) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  struct entry_data_t {
    iree_atomic_int32_t value;
  } entry_data;
  iree_atomic_store(&entry_data.value, 0, iree_memory_order_relaxed);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Push a few overrides.
  // Some platforms (Apple) may ignore the request and return NULL. Code using
  // overrides needs to be tolerant of this.
  iree_thread_override_t* override0 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGH);
  iree_thread_override_t* override1 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGHEST);
  iree_thread_override_t* override2 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_LOWEST);

  // Wait for the thread to finish.
  while (iree_atomic_load(&entry_data.value, iree_memory_order_acquire) != 1) {
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

  // Out of order to ensure highest bit sticks.
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
