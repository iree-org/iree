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

#include "iree/hal/vulkan/util/intrusive_list.h"

#include <algorithm>
#include <vector>

#include "iree/testing/gtest.h"

namespace iree {
namespace {

using ::testing::ElementsAre;

struct Item {
  size_t some_data_0;
  IntrusiveListLink list_a;
  size_t some_data_1;
  IntrusiveListLink list_b;
  size_t some_data_2;
  int value;

  static constexpr size_t kToken = 0xDEADBEEF;
  explicit Item(int value)
      : some_data_0(kToken),
        some_data_1(kToken),
        some_data_2(kToken),
        value(value) {}
  bool is_valid() {
    return some_data_0 == kToken && some_data_1 == kToken &&
           some_data_2 == kToken;
  }
};

template <typename T, size_t V>
std::vector<T*> ExtractItems(const IntrusiveList<T, V>& list) {
  std::vector<T*> items;
  for (auto* item : list) {
    items.push_back(item);
  }
  return items;
}

template <typename T, size_t V>
std::vector<int> ExtractValues(const IntrusiveList<T, V>& list) {
  std::vector<int> values;
  for (auto* item : list) {
    values.push_back(item->value);
  }
  return values;
}

template <typename T, size_t V>
std::vector<int> ExtractValuesMutable(const IntrusiveList<T, V>& list) {
  std::vector<int> values;
  for (auto* item : list) {
    values.push_back(item->value);
  }
  return values;
}

TEST(IntrusiveListTest, PushPopItems) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  EXPECT_TRUE(items.empty());
  EXPECT_EQ(items.size(), 0u);
  EXPECT_EQ(items.front(), nullptr);
  EXPECT_EQ(items.back(), nullptr);
  EXPECT_TRUE(items.begin() == items.end());
  items.push_front(&item1);
  EXPECT_FALSE(items.empty());
  EXPECT_EQ(items.size(), 1u);
  EXPECT_EQ(items.front(), &item1);
  EXPECT_EQ(items.back(), &item1);
  EXPECT_FALSE(items.begin() == items.end());
  items.push_front(&item2);
  EXPECT_EQ(items.size(), 2u);
  EXPECT_EQ(items.front(), &item2);
  EXPECT_EQ(items.back(), &item1);
  items.push_front(&item3);
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items.front(), &item3);
  EXPECT_EQ(items.back(), &item1);
  EXPECT_THAT(ExtractValues(items), ElementsAre(3, 2, 1));

  items.push_back(&item4);
  EXPECT_EQ(items.size(), 4u);
  EXPECT_EQ(items.front(), &item3);
  EXPECT_EQ(items.back(), &item4);
  EXPECT_THAT(ExtractValues(items), ElementsAre(3, 2, 1, 4));

  items.pop_front();
  EXPECT_EQ(items.size(), 3u);
  EXPECT_EQ(items.front(), &item2);
  EXPECT_EQ(items.back(), &item4);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 1, 4));

  items.pop_back();
  EXPECT_EQ(items.size(), 2u);
  EXPECT_EQ(items.front(), &item2);
  EXPECT_EQ(items.back(), &item1);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 1));

  items.pop_back();
  items.pop_front();
  EXPECT_TRUE(items.empty());
  EXPECT_EQ(items.size(), 0u);
  EXPECT_EQ(items.front(), nullptr);
  EXPECT_EQ(items.back(), nullptr);
  EXPECT_TRUE(items.begin() == items.end());

  EXPECT_TRUE(item1.is_valid());
  EXPECT_TRUE(item2.is_valid());
  EXPECT_TRUE(item3.is_valid());
  EXPECT_TRUE(item4.is_valid());
}

TEST(IntrusiveListTest, Contains) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  // item4 omitted.

  EXPECT_TRUE(items.contains(&item1));
  EXPECT_TRUE(items.contains(&item2));
  EXPECT_TRUE(items.contains(&item3));
  EXPECT_FALSE(items.contains(&item4));

  EXPECT_FALSE(items.contains(nullptr));
}

TEST(IntrusiveListTest, MergeFrom) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items0;
  items0.push_back(&item1);
  items0.push_back(&item2);
  items0.push_back(&item3);

  IntrusiveList<Item, offsetof(Item, list_a)> items1;
  items1.push_back(&item4);

  items0.merge_from(&items1);
  EXPECT_THAT(ExtractValues(items0), ElementsAre(1, 2, 3, 4));
  EXPECT_TRUE(items1.empty());
}

TEST(IntrusiveListTest, MergeFromEmpty) {
  IntrusiveList<Item, offsetof(Item, list_a)> items0;
  IntrusiveList<Item, offsetof(Item, list_a)> items1;
  items0.merge_from(&items1);
}

TEST(IntrusiveListTest, MergeFromAll) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);
  IntrusiveList<Item, offsetof(Item, list_a)> items0;
  items0.push_back(&item1);
  items0.push_back(&item2);
  items0.push_back(&item3);
  items0.push_back(&item4);
  IntrusiveList<Item, offsetof(Item, list_a)> items1;

  // Merge all items from items1 into items0. Shouldn't change anything.
  items0.merge_from(&items1);
  EXPECT_THAT(ExtractValues(items0), ElementsAre(1, 2, 3, 4));
  EXPECT_TRUE(items1.empty());

  // Merge all items from items0 into items1. Should move everything.
  items1.merge_from(&items0);
  EXPECT_TRUE(items0.empty());
  EXPECT_THAT(ExtractValues(items1), ElementsAre(1, 2, 3, 4));
}

TEST(IntrusiveListTest, Erase) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  items.push_back(&item4);

  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));
  items.erase(&item3);
  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 4));
  items.erase(&item1);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 4));
  items.erase(&item4);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2));
  items.erase(&item2);
  EXPECT_TRUE(items.empty());

  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  items.push_back(&item4);

  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));
  auto it = items.begin();
  items.erase(it);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 3, 4));
  it = items.end();
  items.erase(it);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 3, 4));
  it = items.begin();
  ++it;
  items.erase(it);
  EXPECT_THAT(ExtractValues(items), ElementsAre(2, 4));

  it = items.begin();
  it = items.erase(it);
  EXPECT_EQ(4, (*it)->value);
  EXPECT_THAT(ExtractValues(items), ElementsAre(4));
  it = items.erase(it);
  EXPECT_TRUE(items.empty());
  EXPECT_EQ(items.end(), it);
}

TEST(IntrusiveListTest, MultipleLists) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items_a;
  IntrusiveList<Item, offsetof(Item, list_b)> items_b;
  items_a.push_back(&item1);
  items_a.push_back(&item2);
  items_a.push_back(&item3);
  items_a.push_back(&item4);
  items_b.push_front(&item1);
  items_b.push_front(&item2);
  items_b.push_front(&item3);
  items_b.push_front(&item4);
  EXPECT_THAT(ExtractValues(items_a), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(ExtractValues(items_b), ElementsAre(4, 3, 2, 1));
  items_b.erase(&item3);
  EXPECT_THAT(ExtractValues(items_a), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(ExtractValues(items_b), ElementsAre(4, 2, 1));
  items_a.pop_back();
  EXPECT_THAT(ExtractValues(items_a), ElementsAre(1, 2, 3));
  EXPECT_THAT(ExtractValues(items_b), ElementsAre(4, 2, 1));
}

TEST(IntrusiveListTest, MutableIterator) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.push_back(&item4);
  items.push_front(&item1);
  items.push_front(&item2);
  items.push_front(&item3);

  EXPECT_THAT(ExtractValuesMutable(items), ElementsAre(3, 2, 1, 4));
}

struct BaseType {
  explicit BaseType(int value) : value(value) {}
  int value;
  IntrusiveListLink base_link;
};
struct SubType : public BaseType {
  explicit SubType(int value) : BaseType(value) {}
  IntrusiveListLink sub_link;
};
TEST(IntrusiveListTest, SimpleType) {
  SubType item1(1);
  SubType item2(2);
  SubType item3(3);
  SubType item4(4);

  IntrusiveList<BaseType, offsetof(BaseType, base_link)> items_a;
  items_a.push_front(&item1);
  items_a.push_front(&item2);
  items_a.push_front(&item3);
  items_a.push_front(&item4);
  EXPECT_THAT(ExtractValues(items_a), ElementsAre(4, 3, 2, 1));

  IntrusiveList<SubType, offsetof(SubType, sub_link)> items_b;
  items_b.push_back(&item1);
  items_b.push_back(&item2);
  items_b.push_back(&item3);
  items_b.push_back(&item4);
  EXPECT_THAT(ExtractValues(items_b), ElementsAre(1, 2, 3, 4));
}

struct AbstractType {
  explicit AbstractType(int value) : value(value) {}
  virtual ~AbstractType() = default;
  virtual int DoSomething() = 0;
  int value;
  IntrusiveListLink base_link;
};
struct ImplType : public AbstractType {
  explicit ImplType(int value) : AbstractType(value) {}
  int DoSomething() override { return value; }
  IntrusiveListLink sub_link;
};

TEST(IntrusiveListTest, ComplexType) {
  ImplType item1(1);
  ImplType item2(2);
  ImplType item3(3);
  ImplType item4(4);

  IntrusiveList<AbstractType, offsetof(AbstractType, base_link)> items_a;
  items_a.push_front(&item1);
  items_a.push_front(&item2);
  items_a.push_front(&item3);
  items_a.push_front(&item4);
  EXPECT_THAT(ExtractValues(items_a), ElementsAre(4, 3, 2, 1));

  IntrusiveList<ImplType, offsetof(ImplType, sub_link)> items_b;
  items_b.push_back(&item1);
  items_b.push_back(&item2);
  items_b.push_back(&item3);
  items_b.push_back(&item4);
  EXPECT_THAT(ExtractValues(items_b), ElementsAre(1, 2, 3, 4));
}

bool Comparison(Item* a, Item* b) { return a->value < b->value; }

TEST(IntrusiveListTest, Inserting) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.insert(items.end(), &item3);
  items.insert(items.begin(), &item1);
  items.insert(items.end(), &item4);

  auto pos = std::upper_bound(items.begin(), items.end(), &item2, Comparison);
  items.insert(pos, &item2);

  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));
}

TEST(IntrusiveListTest, Iteration) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  items.push_back(&item4);

  std::vector<int> regular;
  for (auto it = items.begin(); it != items.end(); ++it) {
    regular.push_back((*it)->value);
  }
  EXPECT_THAT(regular, ElementsAre(1, 2, 3, 4));

  std::vector<int> reverse;
  for (auto rit = items.rbegin(); rit != items.rend(); ++rit) {
    reverse.push_back((*rit)->value);
  }
  EXPECT_THAT(reverse, ElementsAre(4, 3, 2, 1));
}

TEST(IntrusiveListTest, NextPrevious) {
  Item item1(1);
  Item item2(2);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  EXPECT_EQ(nullptr, items.previous(nullptr));
  EXPECT_EQ(nullptr, items.next(nullptr));

  items.push_back(&item1);
  EXPECT_EQ(nullptr, items.previous(&item1));
  EXPECT_EQ(nullptr, items.next(&item1));

  items.push_back(&item2);
  EXPECT_EQ(nullptr, items.previous(&item1));
  EXPECT_EQ(&item2, items.next(&item1));
  EXPECT_EQ(&item1, items.previous(&item2));
  EXPECT_EQ(nullptr, items.next(&item2));
}

TEST(IntrusiveListTest, Clear) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;

  // Empty clear.
  items.clear();
  EXPECT_TRUE(items.empty());

  // 1 item clear.
  items.push_back(&item1);
  items.clear();
  EXPECT_TRUE(items.empty());

  // Multi-item clear.
  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  items.push_back(&item4);
  items.clear();
  EXPECT_TRUE(items.empty());
}

TEST(IntrusiveListTest, ClearDeleter) {
  Item item1(1);
  Item item2(2);

  IntrusiveList<Item, offsetof(Item, list_a)> items;

  // No-op first.
  int delete_count = 0;
  items.clear([&](Item* item) { ++delete_count; });
  EXPECT_EQ(0, delete_count);

  // Now with items.
  items.push_back(&item1);
  items.push_back(&item2);
  items.clear([&](Item* item) { ++delete_count; });
  EXPECT_EQ(2, delete_count);
  EXPECT_TRUE(items.empty());
}

TEST(IntrusiveListTest, Replace) {
  Item item1(1);
  Item item2(2);
  Item item3(3);

  IntrusiveList<Item, offsetof(Item, list_a)> items;
  items.push_back(&item1);
  items.push_back(&item2);

  items.replace(&item1, &item3);
  EXPECT_THAT(ExtractValues(items), ElementsAre(3, 2));
  EXPECT_FALSE(items.contains(&item1));
  items.replace(&item2, &item1);
  EXPECT_THAT(ExtractValues(items), ElementsAre(3, 1));
  EXPECT_FALSE(items.contains(&item2));
}

TEST(IntrusiveListTest, Sort) {
  Item item1(1);
  Item item2(2);
  Item item3(3);
  Item item4(4);

  IntrusiveList<Item, offsetof(Item, list_a)> items;

  // Empty sort.
  items.sort([](Item* a, Item* b) { return a->value < b->value; });

  // Single item sort.
  items.clear();
  items.push_back(&item1);
  items.sort([](Item* a, Item* b) { return a->value < b->value; });
  EXPECT_THAT(ExtractValues(items), ElementsAre(1));

  // Already sorted.
  items.clear();
  items.push_back(&item1);
  items.push_back(&item2);
  items.push_back(&item3);
  items.push_back(&item4);
  items.sort([](Item* a, Item* b) { return a->value < b->value; });
  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));

  // Reverse.
  items.clear();
  items.push_back(&item4);
  items.push_back(&item3);
  items.push_back(&item2);
  items.push_back(&item1);
  items.sort([](Item* a, Item* b) { return a->value < b->value; });
  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));

  // Random.
  items.clear();
  items.push_back(&item2);
  items.push_back(&item4);
  items.push_back(&item1);
  items.push_back(&item3);
  items.sort([](Item* a, Item* b) { return a->value < b->value; });
  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 2, 3, 4));

  // Stability.
  Item item1a(1);
  Item item2a(2);
  items.clear();
  items.push_back(&item2);
  items.push_back(&item4);
  items.push_back(&item1);
  items.push_back(&item3);
  items.push_back(&item1a);
  items.push_back(&item2a);
  items.sort([](Item* a, Item* b) { return a->value <= b->value; });
  EXPECT_THAT(ExtractValues(items), ElementsAre(1, 1, 2, 2, 3, 4));
  auto items_vector = ExtractItems(items);
  EXPECT_EQ(&item1, items_vector[0]);
  EXPECT_EQ(&item1a, items_vector[1]);
  EXPECT_EQ(&item2, items_vector[2]);
  EXPECT_EQ(&item2a, items_vector[3]);
  items.clear();
}

}  // namespace
}  // namespace iree
