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

#include "iree/hal/vulkan/util/arena.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace {

// Tests basic block allocations.
TEST(ArenaTest, BasicAllocation) {
  Arena arena(64);
  EXPECT_EQ(64, arena.block_size());
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  // Zero byte allocations should return nullptr and not allocate bytes.
  auto zero_ptr = reinterpret_cast<uintptr_t>(arena.AllocateBytes(0));
  EXPECT_EQ(0, zero_ptr);
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  arena.Clear();

  // Allocations must be machine word aligned.
  auto one_ptr = reinterpret_cast<uintptr_t>(arena.AllocateBytes(1));
  EXPECT_NE(0, one_ptr);
  EXPECT_EQ(0, one_ptr % sizeof(uintptr_t));
  one_ptr = reinterpret_cast<uintptr_t>(arena.AllocateBytes(1));
  EXPECT_NE(0, one_ptr);
  EXPECT_EQ(0, one_ptr % sizeof(uintptr_t));
  EXPECT_EQ(2, arena.bytes_allocated());
  EXPECT_LT(2, arena.block_bytes_allocated());

  arena.Clear();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());
}

// Tests typed allocations.
TEST(ArenaTest, TypedAllocations) {
  Arena arena(64);

  EXPECT_NE(nullptr, arena.Allocate<int>());
  EXPECT_EQ(4, arena.bytes_allocated());
  EXPECT_EQ(64 + Arena::kBlockOverhead, arena.block_bytes_allocated());
  arena.Clear();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  struct MyType {
    MyType() {}
    explicit MyType(int initial_value) : value(initial_value) {}

    int value = 5;
  };
  auto my_type_ptr = arena.Allocate<MyType>();
  EXPECT_NE(nullptr, my_type_ptr);
  EXPECT_EQ(sizeof(MyType), arena.bytes_allocated());
  EXPECT_EQ(5, my_type_ptr->value);  // Default ctor must be called.
  arena.Clear();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  my_type_ptr = arena.Allocate<MyType>(10);
  EXPECT_NE(nullptr, my_type_ptr);
  EXPECT_EQ(sizeof(MyType), arena.bytes_allocated());
  EXPECT_EQ(10, my_type_ptr->value);  // Ctor should have been called.
  arena.Clear();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());
}

// Tests multiple blocks.
TEST(ArenaTest, MultipleBlocks) {
  Arena arena(16);
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  // Allocate one entire block.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(16, arena.bytes_allocated());
  EXPECT_EQ(16 + Arena::kBlockOverhead, arena.block_bytes_allocated());

  // Allocate into the next block.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(32, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());

  // Clear.
  arena.Clear();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  // Allocate again.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(16, arena.bytes_allocated());
  EXPECT_EQ(16 + Arena::kBlockOverhead, arena.block_bytes_allocated());
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(32, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());
}

// Tests fast reset.
TEST(ArenaTest, FastReset) {
  Arena arena(16);
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(0, arena.block_bytes_allocated());

  // Allocate one entire block.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(16, arena.bytes_allocated());
  EXPECT_EQ(16 + Arena::kBlockOverhead, arena.block_bytes_allocated());

  // Allocate into the next block.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(32, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());

  // Reset (without deallocating).
  arena.Reset();
  EXPECT_EQ(0, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());

  // Allocate again.
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(16, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());
  EXPECT_NE(nullptr, arena.AllocateBytes(16));
  EXPECT_EQ(32, arena.bytes_allocated());
  EXPECT_EQ(32 + 2 * Arena::kBlockOverhead, arena.block_bytes_allocated());
}

}  // namespace
}  // namespace iree
