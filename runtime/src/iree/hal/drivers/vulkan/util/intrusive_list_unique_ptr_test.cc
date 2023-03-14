// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/hal/drivers/vulkan/util/intrusive_list.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace {

struct AllocatedType : public IntrusiveLinkBase<void> {
  AllocatedType() { ++alloc_count; }
  ~AllocatedType() { --alloc_count; }
  static int alloc_count;
};
int AllocatedType::alloc_count = 0;

TEST(IntrusiveListUniquePtrTest, UniquePtr) {
  AllocatedType::alloc_count = 0;

  // Push/clear.
  IntrusiveList<std::unique_ptr<AllocatedType>> list;
  EXPECT_EQ(0, AllocatedType::alloc_count);
  list.push_back(std::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  list.clear();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/pop.
  list.push_back(std::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  for (auto item : list) {
    EXPECT_EQ(item, list.front());
  }
  list.pop_back();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/take.
  list.push_back(std::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  auto item = list.take(list.front());
  EXPECT_TRUE(list.empty());
  EXPECT_NE(nullptr, item.get());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  item.reset();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/replace.
  list.push_back(std::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  list.replace(list.front(), std::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  list.clear();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Iteration.
  list.push_back(std::make_unique<AllocatedType>());
  list.push_back(std::make_unique<AllocatedType>());
  list.push_back(std::make_unique<AllocatedType>());
  EXPECT_EQ(3, AllocatedType::alloc_count);
  for (auto item : list) {
    AllocatedType* item_ptr = item;
    EXPECT_NE(nullptr, item_ptr);
  }
  list.clear();
  EXPECT_EQ(0, AllocatedType::alloc_count);
}

}  // namespace
}  // namespace iree
