// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/synchronization.h"

#include <chrono>
#include <thread>

#include "iree/testing/gtest.h"

namespace {

//==============================================================================
// Test utils
//==============================================================================

template <typename T>
class Mutex;

template <>
class Mutex<iree_mutex_t> {
 public:
  static void Initialize(iree_mutex_t* out_mu) {
    iree_mutex_initialize(out_mu);
  }
  static void Deinitialize(iree_mutex_t* mu) { iree_mutex_deinitialize(mu); }
  static void Lock(iree_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_mutex_lock(mu);
  }
  static bool TryLock(iree_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    return iree_mutex_try_lock(mu);
  }
  static void Unlock(iree_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_mutex_unlock(mu);
  }
};

template <>
class Mutex<iree_slim_mutex_t> {
 public:
  static void Initialize(iree_slim_mutex_t* out_mu) {
    iree_slim_mutex_initialize(out_mu);
  }
  static void Deinitialize(iree_slim_mutex_t* mu) {
    iree_slim_mutex_deinitialize(mu);
  }
  static void Lock(iree_slim_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_slim_mutex_lock(mu);
  }
  static bool TryLock(iree_slim_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    return iree_slim_mutex_try_lock(mu);
  }
  static void Unlock(iree_slim_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_slim_mutex_unlock(mu);
  }
};

// Tests that a mutex allows exclusive access to a region by touching it from
// multiple threads.
template <typename T>
void TestMutexExclusiveAccess() {
  // We'll increment the counter back and forth as we touch it from multiple
  // threads.
  int counter = 0;

  T mu;
  Mutex<T>::Initialize(&mu);

  // Hold the lock at the start. The threads should block waiting for the lock
  // to be released so they can take it.
  ASSERT_EQ(0, counter);
  Mutex<T>::Lock(&mu);

  // Start up a thread to ++counter (it should block since we hold the lock).
  std::thread th1([&]() {
    Mutex<T>::Lock(&mu);
    ++counter;
    Mutex<T>::Unlock(&mu);
  });

  // Unlock and wait for the thread to acquire the lock and finish its work.
  ASSERT_EQ(0, counter);
  Mutex<T>::Unlock(&mu);
  th1.join();

  // Thread should have been able to increment the counter.
  ASSERT_EQ(1, counter);

  Mutex<T>::Deinitialize(&mu);
}

// Tests that try lock bails when the lock is held by another thread.
template <typename T>
void TestMutexExclusiveAccessTryLock() {
  int counter = 0;
  T mu;
  Mutex<T>::Initialize(&mu);

  // Hold the lock at the start. The try lock should fail and the thread should
  // exit without changing the counter value.
  ASSERT_EQ(0, counter);
  Mutex<T>::Lock(&mu);
  std::thread th1([&]() {
    if (Mutex<T>::TryLock(&mu)) {
      ++counter;
      Mutex<T>::Unlock(&mu);
    }
  });

  // Wait for the thread to try (and fail).
  th1.join();
  Mutex<T>::Unlock(&mu);

  // The thread should not have been able to change the counter.
  ASSERT_EQ(0, counter);

  Mutex<T>::Deinitialize(&mu);
}

//==============================================================================
// iree_mutex_t
//==============================================================================

TEST(MutexTest, Lifetime) {
  iree_mutex_t mutex;
  iree_mutex_initialize(&mutex);
  bool did_lock = iree_mutex_try_lock(&mutex);
  EXPECT_TRUE(did_lock);
  if (did_lock) iree_mutex_unlock(&mutex);
  iree_mutex_lock(&mutex);
  iree_mutex_unlock(&mutex);
  iree_mutex_deinitialize(&mutex);
}

TEST(MutexTest, ExclusiveAccess) { TestMutexExclusiveAccess<iree_mutex_t>(); }

TEST(MutexTest, ExclusiveAccessTryLock) {
  TestMutexExclusiveAccessTryLock<iree_mutex_t>();
}

//==============================================================================
// iree_slim_mutex_t
//==============================================================================

TEST(SlimMutexTest, Lifetime) {
  iree_slim_mutex_t mutex;
  iree_slim_mutex_initialize(&mutex);
  bool did_lock = iree_slim_mutex_try_lock(&mutex);
  EXPECT_TRUE(did_lock);
  if (did_lock) iree_slim_mutex_unlock(&mutex);
  iree_slim_mutex_lock(&mutex);
  iree_slim_mutex_unlock(&mutex);
  iree_slim_mutex_deinitialize(&mutex);
}

TEST(SlimMutexTest, ExclusiveAccess) {
  TestMutexExclusiveAccess<iree_slim_mutex_t>();
}

TEST(SlimMutexTest, ExclusiveAccessTryLock) {
  TestMutexExclusiveAccessTryLock<iree_slim_mutex_t>();
}

//==============================================================================
// iree_notification_t
//==============================================================================

// Tested implicitly in threading_test.cc.

}  // namespace
