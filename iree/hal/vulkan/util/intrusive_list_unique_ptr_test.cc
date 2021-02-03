// Copyright 2019 Google LLC
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

#include "absl/memory/memory.h"
#include "iree/hal/vulkan/util/intrusive_list.h"
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
  list.push_back(absl::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  list.clear();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/pop.
  list.push_back(absl::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  for (auto item : list) {
    EXPECT_EQ(item, list.front());
  }
  list.pop_back();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/take.
  list.push_back(absl::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  EXPECT_NE(nullptr, list.front());
  auto item = list.take(list.front());
  EXPECT_TRUE(list.empty());
  EXPECT_NE(nullptr, item.get());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  item.reset();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Push/replace.
  list.push_back(absl::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  list.replace(list.front(), absl::make_unique<AllocatedType>());
  EXPECT_EQ(1, AllocatedType::alloc_count);
  list.clear();
  EXPECT_EQ(0, AllocatedType::alloc_count);

  // Iteration.
  list.push_back(absl::make_unique<AllocatedType>());
  list.push_back(absl::make_unique<AllocatedType>());
  list.push_back(absl::make_unique<AllocatedType>());
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
