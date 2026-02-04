// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/call_once.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

TEST(CallOnceTest, SingleInvocation) {
  static int counter = 0;
  static iree_once_flag flag = IREE_ONCE_FLAG_INIT;

  auto init_fn = +[]() { ++counter; };

  iree_call_once(&flag, init_fn);
  iree_call_once(&flag, init_fn);
  iree_call_once(&flag, init_fn);

  EXPECT_EQ(1, counter);
}

TEST(CallOnceTest, ConcurrentCalls) {
  static std::atomic<int> concurrent_counter{0};
  static iree_once_flag concurrent_flag = IREE_ONCE_FLAG_INIT;

  auto init_fn = +[]() { concurrent_counter.fetch_add(1); };

  std::vector<std::thread> threads;
  constexpr int kNumThreads = 8;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() { iree_call_once(&concurrent_flag, init_fn); });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(1, concurrent_counter.load());
}

TEST(CallOnceTest, MultipleFlags) {
  static int counter_a = 0;
  static int counter_b = 0;
  static iree_once_flag flag_a = IREE_ONCE_FLAG_INIT;
  static iree_once_flag flag_b = IREE_ONCE_FLAG_INIT;

  auto init_a = +[]() { ++counter_a; };
  auto init_b = +[]() { ++counter_b; };

  iree_call_once(&flag_a, init_a);
  iree_call_once(&flag_b, init_b);
  iree_call_once(&flag_a, init_a);
  iree_call_once(&flag_b, init_b);

  EXPECT_EQ(1, counter_a);
  EXPECT_EQ(1, counter_b);
}

TEST(CallOnceTest, ConcurrentDifferentFlags) {
  static std::atomic<int> counter_x{0};
  static std::atomic<int> counter_y{0};
  static iree_once_flag flag_x = IREE_ONCE_FLAG_INIT;
  static iree_once_flag flag_y = IREE_ONCE_FLAG_INIT;

  auto init_x = +[]() { counter_x.fetch_add(1); };
  auto init_y = +[]() { counter_y.fetch_add(1); };

  std::vector<std::thread> threads;
  constexpr int kNumThreads = 8;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      if (i % 2 == 0) {
        iree_call_once(&flag_x, init_x);
      } else {
        iree_call_once(&flag_y, init_y);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(1, counter_x.load());
  EXPECT_EQ(1, counter_y.load());
}

}  // namespace
