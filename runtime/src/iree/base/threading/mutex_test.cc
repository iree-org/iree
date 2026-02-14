// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/mutex.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

//==============================================================================
// Test utilities
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

// Tests high contention with multiple threads.
template <typename T>
void TestMutexHighContention() {
  constexpr int kNumThreads = 8;
  constexpr int kIncrementsPerThread = 1000;

  std::atomic<int> counter{0};
  T mu;
  Mutex<T>::Initialize(&mu);

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < kIncrementsPerThread; ++j) {
        Mutex<T>::Lock(&mu);
        counter.fetch_add(1, std::memory_order_relaxed);
        Mutex<T>::Unlock(&mu);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(kNumThreads * kIncrementsPerThread,
            counter.load(std::memory_order_acquire));

  Mutex<T>::Deinitialize(&mu);
}

// Tests try_lock under contention.
template <typename T>
void TestMutexTryLockContended() {
  T mu;
  Mutex<T>::Initialize(&mu);

  // Hold lock in main thread.
  Mutex<T>::Lock(&mu);

  constexpr int kNumThreads = 4;
  std::atomic<int> attempted_count{0};
  std::atomic<int> failed_count{0};
  std::atomic<int> succeeded_count{0};
  std::atomic<bool> main_released{false};

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      // First try should fail since main holds the lock.
      if (!Mutex<T>::TryLock(&mu)) {
        failed_count.fetch_add(1, std::memory_order_relaxed);
      } else {
        Mutex<T>::Unlock(&mu);
      }
      // Signal that we've attempted our first try_lock.
      attempted_count.fetch_add(1, std::memory_order_release);

      // Wait for main to release.
      while (!main_released.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      // Try again - one should succeed.
      if (Mutex<T>::TryLock(&mu)) {
        succeeded_count.fetch_add(1, std::memory_order_relaxed);
        // Hold briefly to let others fail.
        std::this_thread::yield();
        Mutex<T>::Unlock(&mu);
      }
    });
  }

  // Wait for all threads to attempt their first try_lock.
  while (attempted_count.load(std::memory_order_acquire) < kNumThreads) {
    std::this_thread::yield();
  }

  // All threads should have failed their first try.
  EXPECT_EQ(kNumThreads, failed_count.load(std::memory_order_acquire));

  // Release and let threads race.
  Mutex<T>::Unlock(&mu);
  main_released.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  // At least one thread should have succeeded.
  EXPECT_GE(succeeded_count.load(std::memory_order_acquire), 1);

  Mutex<T>::Deinitialize(&mu);
}

//==============================================================================
// iree_mutex_t
//==============================================================================

TEST(MutexTest, Lifetime) {
  iree_mutex_t mutex;
  iree_mutex_initialize(&mutex);
  while (!iree_mutex_try_lock(&mutex)) {
    // Functions with try in their name may fail spuriously.
  }
  iree_mutex_unlock(&mutex);
  iree_mutex_lock(&mutex);
  iree_mutex_unlock(&mutex);
  iree_mutex_deinitialize(&mutex);
}

TEST(MutexTest, ExclusiveAccess) { TestMutexExclusiveAccess<iree_mutex_t>(); }

TEST(MutexTest, ExclusiveAccessTryLock) {
  TestMutexExclusiveAccessTryLock<iree_mutex_t>();
}

TEST(MutexTest, HighContention) { TestMutexHighContention<iree_mutex_t>(); }

TEST(MutexTest, TryLockContended) { TestMutexTryLockContended<iree_mutex_t>(); }

//==============================================================================
// iree_slim_mutex_t
//==============================================================================

TEST(SlimMutexTest, Lifetime) {
  iree_slim_mutex_t mutex;
  iree_slim_mutex_initialize(&mutex);
  while (!iree_slim_mutex_try_lock(&mutex)) {
    // Functions with try in their name may fail spuriously.
  }
  iree_slim_mutex_unlock(&mutex);
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

TEST(SlimMutexTest, HighContention) {
  TestMutexHighContention<iree_slim_mutex_t>();
}

TEST(SlimMutexTest, TryLockContended) {
  TestMutexTryLockContended<iree_slim_mutex_t>();
}

}  // namespace
