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

#include "base/intrusive_list.h"
#include "gtest/gtest.h"

namespace iree {
namespace {

static int alloc_count = 0;
struct RefCountedType : public RefObject<RefCountedType> {
  IntrusiveListLink link;
  RefCountedType() { ++alloc_count; }
  ~RefCountedType() { --alloc_count; }
  static void Deallocate(RefCountedType* value) { delete value; }
  using RefObject<RefCountedType>::counter_;
};

TEST(IntrusiveListRefPtrTest, PushAndClear) {
  alloc_count = 0;
  IntrusiveList<ref_ptr<RefCountedType>> list;
  EXPECT_EQ(0, alloc_count);
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(1, alloc_count);
  EXPECT_NE(nullptr, list.front());
  EXPECT_EQ(2, list.front()->counter_);
  list.clear();
  EXPECT_EQ(0, alloc_count);
}

TEST(IntrusiveListRefPtrTest, PushPop) {
  alloc_count = 0;
  IntrusiveList<ref_ptr<RefCountedType>> list;
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(1, alloc_count);
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(2, alloc_count);
  EXPECT_NE(list.front(), list.back());
  list.pop_back();
  EXPECT_EQ(1, alloc_count);
  list.pop_front();
  EXPECT_EQ(0, alloc_count);
}

TEST(IntrusiveListRefPtrTest, PushErase) {
  alloc_count = 0;
  IntrusiveList<ref_ptr<RefCountedType>> list;
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(1, alloc_count);
  EXPECT_NE(nullptr, list.front());
  EXPECT_EQ(2, list.front()->counter_);
  auto item = list.front();
  EXPECT_NE(nullptr, item.get());
  EXPECT_EQ(3, list.front()->counter_);
  EXPECT_EQ(1, alloc_count);
  list.erase(item);
  EXPECT_EQ(1, alloc_count);
  item.reset();
  EXPECT_EQ(0, alloc_count);
}

TEST(IntrusiveListRefPtrTest, PushReplace) {
  alloc_count = 0;
  IntrusiveList<ref_ptr<RefCountedType>> list;
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(1, alloc_count);
  list.replace(list.front(), make_ref<RefCountedType>());
  EXPECT_EQ(1, alloc_count);
  list.clear();
  EXPECT_EQ(0, alloc_count);
}

TEST(IntrusiveListRefPtrTest, Iteration) {
  alloc_count = 0;
  IntrusiveList<ref_ptr<RefCountedType>> list;
  list.push_back(make_ref<RefCountedType>());
  list.push_back(make_ref<RefCountedType>());
  list.push_back(make_ref<RefCountedType>());
  EXPECT_EQ(3, alloc_count);
  for (auto item : list) {
    const ref_ptr<RefCountedType>& item_ref = item;
    EXPECT_NE(nullptr, item_ref.get());
  }
  list.clear();
  EXPECT_EQ(0, alloc_count);
}

}  // namespace
}  // namespace iree
