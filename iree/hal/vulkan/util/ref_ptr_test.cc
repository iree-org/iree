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

#include "iree/hal/vulkan/util/ref_ptr.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace {

class MyType : public RefObject<MyType> {
 public:
  int x = 5;

  using RefObject<MyType>::counter_;  // Expose for testing.
};

TEST(RefPtrTest, Construction) {
  // Empty.
  ref_ptr<MyType> n1;
  EXPECT_EQ(nullptr, n1.get());
  ref_ptr<MyType> n2(nullptr);
  EXPECT_EQ(nullptr, n2.get());

  // Assign a new ptr and add ref.
  MyType* a_ptr = new MyType();
  EXPECT_EQ(1, a_ptr->counter_);
  ref_ptr<MyType> a(a_ptr);
  EXPECT_EQ(1, a->counter_);

  // Assign existing ptr without adding a ref.
  ref_ptr<MyType> b(a_ptr);
  EXPECT_EQ(1, b->counter_);

  // Add a new ref.
  ref_ptr<MyType> c = add_ref(b);
  EXPECT_EQ(2, c->counter_);

  b.release();
}

TEST(RefPtrTest, Assign) {
  // Ok to assign nothing.
  ref_ptr<MyType> n1 = assign_ref<MyType>(nullptr);
  EXPECT_EQ(nullptr, n1.get());

  ref_ptr<MyType> mt = make_ref<MyType>();
  EXPECT_EQ(1, mt->counter_);
  ref_ptr<MyType> n2 = assign_ref(mt.get());
  EXPECT_EQ(1, mt->counter_);
  mt.release();  // must release, as we assigned to n2.
  EXPECT_EQ(1, n2->counter_);
  n2.reset();
}

TEST(RefPtrTest, Retain) {
  // Ok to retain nothing.
  ref_ptr<MyType> n1 = add_ref<MyType>(nullptr);
  EXPECT_EQ(nullptr, n1.get());

  ref_ptr<MyType> mt = make_ref<MyType>();
  EXPECT_EQ(1, mt->counter_);
  ref_ptr<MyType> n2 = add_ref(mt.get());
  EXPECT_EQ(2, mt->counter_);
  mt.reset();
  EXPECT_EQ(1, n2->counter_);
  n2.reset();
}

TEST(RefPtrTest, Reset) {
  ref_ptr<MyType> a(new MyType());
  ref_ptr<MyType> b(new MyType());

  // Reset to drop reference.
  ref_ptr<MyType> a_copy = add_ref(a);
  EXPECT_EQ(2, a_copy->counter_);
  a.reset();
  EXPECT_EQ(1, a_copy->counter_);

  // Reset via = operator.
  a = nullptr;
  EXPECT_EQ(1, a_copy->counter_);
  a = add_ref(a_copy);
  EXPECT_EQ(2, a_copy->counter_);

  // No-op on empty ptrs.
  ref_ptr<MyType> n;
  n.reset();
  n.assign(nullptr);
}

TEST(RefPtrTest, ReleaseAssign) {
  ref_ptr<MyType> a(new MyType());

  // Release a's pointer.
  MyType* a_raw_ptr = a.get();
  MyType* a_ptr = a.release();
  EXPECT_EQ(a_raw_ptr, a_ptr);
  EXPECT_EQ(nullptr, a.get());
  EXPECT_EQ(1, a_ptr->counter_);

  // Re-wrap in a ref_ptr.
  a.assign(a_ptr);
  EXPECT_EQ(1, a->counter_);

  // No-op on empty ptrs.
  ref_ptr<MyType> n;
  EXPECT_EQ(nullptr, n.release());
}

TEST(RefPtrTest, Accessors) {
  ref_ptr<MyType> a(new MyType());
  EXPECT_EQ(5, a->x);
  a->x = 100;
  EXPECT_EQ(100, a->x);

  MyType& ra = *a;
  ra.x = 200;
  EXPECT_EQ(200, ra.x);

  const MyType& cra = *a;
  EXPECT_EQ(200, cra.x);
}

TEST(RefPtrTest, BooleanExpressions) {
  ref_ptr<MyType> a(new MyType());
  ref_ptr<MyType> n;

  EXPECT_NE(nullptr, a.get());
  EXPECT_TRUE(a);
  EXPECT_FALSE(!a);
  EXPECT_EQ(true, static_cast<bool>(a));

  EXPECT_EQ(nullptr, n.get());
  EXPECT_FALSE(n);
  EXPECT_TRUE(!n);
  EXPECT_EQ(false, static_cast<bool>(n));
}

TEST(RefPtrTest, Comparisons) {
  ref_ptr<MyType> a(new MyType());
  ref_ptr<MyType> b(new MyType());
  ref_ptr<MyType> n;

  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a == a.get());
  EXPECT_TRUE(a.get() == a);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a != a.get());
  EXPECT_FALSE(a.get() != a);

  EXPECT_FALSE(a == b);
  EXPECT_FALSE(a == b.get());
  EXPECT_FALSE(a.get() == b);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a != b.get());
  EXPECT_TRUE(a.get() != b);

  EXPECT_TRUE(n == n);
  EXPECT_TRUE(n == n.get());
  EXPECT_TRUE(n.get() == n);
  EXPECT_FALSE(n != n);
  EXPECT_FALSE(n != n.get());
  EXPECT_FALSE(n.get() != n);

  EXPECT_FALSE(a < a);
  EXPECT_TRUE(n < a);
}

TEST(RefPtrTest, Swap) {
  ref_ptr<MyType> a(new MyType());
  ref_ptr<MyType> b(new MyType());
  MyType* a_ptr = a.get();
  MyType* b_ptr = b.get();

  swap(a, a);
  EXPECT_EQ(a_ptr, a);

  swap(a, b);
  EXPECT_EQ(a_ptr, b.get());
  EXPECT_EQ(b_ptr, a.get());

  swap(a, b);
  EXPECT_EQ(a_ptr, a.get());
  EXPECT_EQ(b_ptr, b.get());

  ref_ptr<MyType> c;
  swap(a, c);
  EXPECT_EQ(a_ptr, c.get());
  EXPECT_EQ(nullptr, a.get());
}

TEST(RefPtrTest, Move) {
  auto a = make_ref<MyType>();
  auto b = make_ref<MyType>();
  ref_ptr<MyType> c;
  EXPECT_EQ(nullptr, c.get());

  c = std::move(a);
  EXPECT_NE(nullptr, c.get());

  b = std::move(c);
  EXPECT_NE(nullptr, b.get());
}

TEST(RefPtrTest, MoveCompatible) {
  struct MyBaseType : public RefObject<MyBaseType> {
    int x = 5;
    using RefObject<MyBaseType>::counter_;  // Expose for testing.

    virtual ~MyBaseType() = default;
  };
  struct MyTypeA : public MyBaseType {
    int a = 6;
  };
  struct MyTypeB : public MyBaseType {
    int b = 7;
  };

  ref_ptr<MyTypeA> a = make_ref<MyTypeA>();
  EXPECT_EQ(1, a->counter_);
  ref_ptr<MyBaseType> base = add_ref(a);
  EXPECT_EQ(a.get(), base.get());
  EXPECT_EQ(2, a->counter_);

  base = make_ref<MyTypeB>();
  EXPECT_EQ(1, a->counter_);
  EXPECT_EQ(1, base->counter_);
}

TEST(RefPtrTest, StackAllocation) {
  static int alloc_count = 0;
  class StackAllocationType : public RefObject<StackAllocationType> {
   public:
    StackAllocationType() { ++alloc_count; }
    ~StackAllocationType() { --alloc_count; }
  };
  {
    StackAllocationType a;
    EXPECT_EQ(1, alloc_count);
  }
  EXPECT_EQ(0, alloc_count);
}

TEST(RefPtrTest, DefaultDeleter) {
  static int alloc_count = 0;
  class DefaultDeleterType : public RefObject<DefaultDeleterType> {
   public:
    DefaultDeleterType() { ++alloc_count; }
    ~DefaultDeleterType() { --alloc_count; }
  };

  // Empty is ok.
  ref_ptr<DefaultDeleterType> n;
  n.reset();

  // Lifecycle.
  EXPECT_EQ(0, alloc_count);
  ref_ptr<DefaultDeleterType> a = make_ref<DefaultDeleterType>();
  EXPECT_EQ(1, alloc_count);
  a.reset();
  EXPECT_EQ(0, alloc_count);
}

TEST(RefPtrTest, InlineDeallocator) {
  static int alloc_count = 0;
  class CustomDeleterType : public RefObject<CustomDeleterType> {
   public:
    CustomDeleterType() { ++alloc_count; }
    static void Delete(CustomDeleterType* ptr) {
      --alloc_count;
      ::operator delete(ptr);
    }
  };

  // Empty is ok.
  ref_ptr<CustomDeleterType> n;
  n.reset();

  // Lifecycle.
  EXPECT_EQ(0, alloc_count);
  auto a = make_ref<CustomDeleterType>();
  EXPECT_EQ(1, alloc_count);
  a.reset();
  EXPECT_EQ(0, alloc_count);
}

class VirtualDtorTypeA : public RefObject<VirtualDtorTypeA> {
 public:
  VirtualDtorTypeA() { ++alloc_count_a; }
  virtual ~VirtualDtorTypeA() { --alloc_count_a; }
  static int alloc_count_a;
};
int VirtualDtorTypeA::alloc_count_a = 0;

class VirtualDtorTypeB : public VirtualDtorTypeA {
 public:
  VirtualDtorTypeB() { ++alloc_count_b; }
  ~VirtualDtorTypeB() override { --alloc_count_b; }
  static int alloc_count_b;
};
int VirtualDtorTypeB::alloc_count_b = 0;

TEST(RefPtrTest, VirtualDestructor) {
  // Empty is ok.
  ref_ptr<VirtualDtorTypeB> n;
  n.reset();

  // Lifecycle.
  EXPECT_EQ(0, VirtualDtorTypeA::alloc_count_a);
  EXPECT_EQ(0, VirtualDtorTypeB::alloc_count_b);
  ref_ptr<VirtualDtorTypeA> a = make_ref<VirtualDtorTypeB>();
  EXPECT_EQ(1, VirtualDtorTypeA::alloc_count_a);
  EXPECT_EQ(1, VirtualDtorTypeB::alloc_count_b);
  a.reset();
  EXPECT_EQ(0, VirtualDtorTypeA::alloc_count_a);
  EXPECT_EQ(0, VirtualDtorTypeB::alloc_count_b);
}

}  // namespace
}  // namespace iree
