// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/list.h"

#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/builtin_types.h"
#include "iree/vm/ref_cc.h"

class A : public iree::vm::RefObject<A> {
 public:
  float data() const { return data_; }
  void set_data(float value) { data_ = value; }

 private:
  float data_ = 1.0f;
};
static iree_vm_ref_type_descriptor_t test_a_descriptor = {0};
IREE_VM_DECLARE_TYPE_ADAPTERS(test_a, A);
IREE_VM_DEFINE_TYPE_ADAPTERS(test_a, A);

class B : public iree::vm::RefObject<B> {
 public:
  int data() const { return data_; }
  void set_data(int value) { data_ = value; }

 private:
  int data_ = 2;
};
static iree_vm_ref_type_descriptor_t test_b_descriptor = {0};
IREE_VM_DECLARE_TYPE_ADAPTERS(test_b, B);
IREE_VM_DEFINE_TYPE_ADAPTERS(test_b, B);

namespace {

using ::iree::Status;
using ::iree::testing::status::StatusIs;

template <typename T>
static void RegisterRefType(iree_vm_ref_type_descriptor_t* descriptor,
                            const char* type_name) {
  if (descriptor->type == IREE_VM_REF_TYPE_NULL) {
    descriptor->type_name = iree_make_cstring_view(type_name);
    descriptor->offsetof_counter = T::offsetof_counter();
    descriptor->destroy = T::DirectDestroy;
    IREE_CHECK_OK(iree_vm_ref_register_type(descriptor));
  }
}

static void RegisterRefTypes() {
  RegisterRefType<A>(&test_a_descriptor, "AType");
  RegisterRefType<B>(&test_b_descriptor, "BType");
}

template <typename T, typename V>
static iree_vm_ref_t MakeRef(V value) {
  iree_vm_ref_t ref = {0};
  auto* obj = new T();
  obj->set_data(value);
  IREE_CHECK_OK(iree_vm_ref_wrap_assign(
      obj, iree::vm::ref_type_descriptor<T>::get()->type, &ref));
  return ref;
}

class VMListTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_vm_register_builtin_types());
    RegisterRefTypes();
  }
};

// Tests simple primitive value list usage, mainly just for demonstration.
// Stores only i32 element types, equivalent to `!vm.list<i32>`.
TEST_F(VMListTest, UsageI32) {
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_value_type(IREE_VM_VALUE_TYPE_I32);
  iree_host_size_t initial_capacity = 123;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));

  iree_vm_type_def_t queried_element_type;
  IREE_ASSERT_OK(iree_vm_list_element_type(list, &queried_element_type));
  EXPECT_TRUE(iree_vm_type_def_is_value(&queried_element_type));
  EXPECT_EQ(0,
            memcmp(&element_type, &queried_element_type, sizeof(element_type)));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));
  EXPECT_EQ(5, iree_vm_list_size(list));

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value = iree_vm_value_make_i32((int32_t)i);
    IREE_ASSERT_OK(iree_vm_list_set_value(list, i, &value));
  }

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value;
    IREE_ASSERT_OK(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I32, &value));
    EXPECT_EQ(IREE_VM_VALUE_TYPE_I32, value.type);
    EXPECT_EQ(i, value.i32);
  }

  iree_vm_list_release(list);
}

// Tests simple ref object list usage, mainly just for demonstration.
// Stores ref object type A elements only, equivalent to `!vm.list<!vm.ref<A>>`.
TEST_F(VMListTest, UsageRef) {
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_ref_type(test_a_type_id());
  iree_host_size_t initial_capacity = 123;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));

  iree_vm_type_def_t queried_element_type;
  IREE_ASSERT_OK(iree_vm_list_element_type(list, &queried_element_type));
  EXPECT_TRUE(iree_vm_type_def_is_ref(&queried_element_type));
  EXPECT_EQ(0,
            memcmp(&element_type, &queried_element_type, sizeof(element_type)));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));
  EXPECT_EQ(5, iree_vm_list_size(list));

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>((float)i);
    IREE_ASSERT_OK(iree_vm_list_set_ref_move(list, i, &ref_a));
  }

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }

  iree_vm_list_release(list);
}

// Tests simple variant list usage, mainly just for demonstration.
// Stores any heterogeneous element type, equivalent to `!vm.list<?>`.
TEST_F(VMListTest, UsageVariant) {
  iree_vm_type_def_t element_type = iree_vm_type_def_make_variant_type();
  iree_host_size_t initial_capacity = 123;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));

  iree_vm_type_def_t queried_element_type;
  IREE_ASSERT_OK(iree_vm_list_element_type(list, &queried_element_type));
  EXPECT_TRUE(iree_vm_type_def_is_variant(&queried_element_type));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  IREE_ASSERT_OK(iree_vm_list_resize(list, 10));
  EXPECT_EQ(10, iree_vm_list_size(list));

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value = iree_vm_value_make_i32((int32_t)i);
    IREE_ASSERT_OK(iree_vm_list_set_value(list, i, &value));
  }
  for (iree_host_size_t i = 5; i < 10; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>(static_cast<float>(i));
    IREE_ASSERT_OK(iree_vm_list_set_ref_move(list, i, &ref_a));
  }

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value;
    IREE_ASSERT_OK(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I32, &value));
    EXPECT_EQ(IREE_VM_VALUE_TYPE_I32, value.type);
    EXPECT_EQ(i, value.i32);
  }
  for (iree_host_size_t i = 5; i < 10; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }

  iree_vm_list_release(list);
}

// Tests capacity reservation.
TEST_F(VMListTest, Reserve) {
  // Allocate with 0 initial capacity (which may get rounded up).
  iree_vm_type_def_t element_type = iree_vm_type_def_make_variant_type();
  iree_host_size_t initial_capacity = 0;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  // Reserve some capacity, which may allocate.
  IREE_ASSERT_OK(iree_vm_list_reserve(list, 100));
  iree_host_size_t current_capacity = iree_vm_list_capacity(list);
  EXPECT_LE(100, current_capacity);

  // Resize to add items, which should not change capacity.
  IREE_ASSERT_OK(iree_vm_list_resize(list, 1));
  EXPECT_EQ(1, iree_vm_list_size(list));
  EXPECT_EQ(current_capacity, iree_vm_list_capacity(list));

  // Reserving <= the current capacity should be a no-op.
  IREE_ASSERT_OK(iree_vm_list_reserve(list, current_capacity));
  EXPECT_EQ(current_capacity, iree_vm_list_capacity(list));

  iree_vm_list_release(list);
}

// Tests the behavior of resize for truncation and extension on primitives.
TEST_F(VMListTest, ResizeI32) {
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_value_type(IREE_VM_VALUE_TYPE_I32);
  iree_host_size_t initial_capacity = 4;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  // Extend and zero-initialize.
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value;
    IREE_ASSERT_OK(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I32, &value));
    EXPECT_EQ(0, value.i32);
  }

  // Overwrite with [0, 5).
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_value_t value = iree_vm_value_make_i32((int32_t)i);
    IREE_ASSERT_OK(iree_vm_list_set_value(list, i, &value));
  }

  // Truncate to [0, 2) and then extend again.
  // This ensures that we test the primitive clearing path during cleanup:
  // [int, int, int, int, int]
  //            |___________| <- truncation region
  IREE_ASSERT_OK(iree_vm_list_resize(list, 2));
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));

  // Ensure that elements 2+ are zeroed after having been reset while 0 and 1
  // are still valid as before.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_vm_value_t value;
    IREE_ASSERT_OK(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I32, &value));
    EXPECT_EQ(i, value.i32);
  }
  for (iree_host_size_t i = 2; i < 5; ++i) {
    iree_vm_value_t value;
    IREE_ASSERT_OK(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I32, &value));
    EXPECT_EQ(0, value.i32);
  }

  iree_vm_list_release(list);
}

// Tests the behavior of resize for truncation and extension on refs.
TEST_F(VMListTest, ResizeRef) {
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_ref_type(test_a_type_id());
  iree_host_size_t initial_capacity = 4;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  // Extend and zero-initialize.
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_assign(list, i, &ref_a));
    EXPECT_TRUE(iree_vm_ref_is_null(&ref_a));
  }

  // Overwrite with [0, 5).
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>((float)i);
    IREE_ASSERT_OK(iree_vm_list_set_ref_move(list, i, &ref_a));
  }

  // Truncate to [0, 2) and then extend again.
  // This ensures that we test the ref path during cleanup:
  // [ref, ref, ref, ref, ref]
  //            |___________| <- truncation region
  IREE_ASSERT_OK(iree_vm_list_resize(list, 2));
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));

  // Ensure that elements 2+ are reset after having been reset while 0 and 1
  // are still valid as before.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }
  for (iree_host_size_t i = 2; i < 5; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_assign(list, i, &ref_a));
    EXPECT_TRUE(iree_vm_ref_is_null(&ref_a));
  }

  iree_vm_list_release(list);
}

// Tests the behavior of resize for truncation and extension on variants.
TEST_F(VMListTest, ResizeVariant) {
  iree_vm_type_def_t element_type = iree_vm_type_def_make_variant_type();
  iree_host_size_t initial_capacity = 4;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  // Extend and zero-initialize.
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_variant_t value = iree_vm_variant_empty();
    IREE_ASSERT_OK(iree_vm_list_get_variant(list, i, &value));
    EXPECT_TRUE(iree_vm_variant_is_empty(value));
  }

  // Overwrite with [0, 5) in mixed types.
  for (iree_host_size_t i = 0; i < 4; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>((float)i);
    IREE_ASSERT_OK(iree_vm_list_set_ref_move(list, i, &ref_a));
  }
  for (iree_host_size_t i = 4; i < 5; ++i) {
    iree_vm_value_t value = iree_vm_value_make_i32((int32_t)i);
    IREE_ASSERT_OK(iree_vm_list_set_value(list, i, &value));
  }

  // Truncate to [0, 2) and then extend again.
  // This ensures that we test the variant path during cleanup:
  // [ref, ref, ref, ref, int]
  //            |___________| <- truncation region
  IREE_ASSERT_OK(iree_vm_list_resize(list, 2));
  IREE_ASSERT_OK(iree_vm_list_resize(list, 5));

  // Ensure that elements 2+ are reset after having been reset while 0 and 1
  // are still valid as before.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }
  for (iree_host_size_t i = 2; i < 5; ++i) {
    iree_vm_variant_t value = iree_vm_variant_empty();
    IREE_ASSERT_OK(iree_vm_list_get_variant(list, i, &value));
    EXPECT_TRUE(iree_vm_variant_is_empty(value));
  }

  iree_vm_list_release(list);
}

// TODO(benvanik): test value get/set.

// TODO(benvanik): test value conversion.

// TODO(benvanik): test ref get/set.

// Tests pushing and popping ref objects.
TEST_F(VMListTest, PushPopRef) {
  iree_vm_type_def_t element_type =
      iree_vm_type_def_make_ref_type(test_a_type_id());
  iree_host_size_t initial_capacity = 4;
  iree_vm_list_t* list = nullptr;
  IREE_ASSERT_OK(iree_vm_list_create(&element_type, initial_capacity,
                                     iree_allocator_system(), &list));
  EXPECT_LE(initial_capacity, iree_vm_list_capacity(list));
  EXPECT_EQ(0, iree_vm_list_size(list));

  // Pops when empty fail.
  iree_vm_ref_t empty_ref{0};
  EXPECT_THAT(Status(iree_vm_list_pop_front_ref_move(list, &empty_ref)),
              StatusIs(iree::StatusCode::kOutOfRange));

  // Push back [0, 5).
  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>((float)i);
    IREE_ASSERT_OK(iree_vm_list_push_ref_move(list, &ref_a));
  }

  // Pop the first two [0, 1] and leave [2, 5).
  // This ensures that we test the ref path during cleanup:
  // [ref, ref, ref, ref, ref]
  //  |______| <- popped region
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_pop_front_ref_move(list, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }

  // Ensure that elements 2+ are valid but now at offset 0.
  for (iree_host_size_t i = 2; i < 5; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i - 2, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }

  // Push back two more to get [2, 7).
  for (iree_host_size_t i = 5; i < 7; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>((float)i);
    IREE_ASSERT_OK(iree_vm_list_push_ref_move(list, &ref_a));
  }

  // Ensure the new elements got added to the end.
  for (iree_host_size_t i = 2; i < 7; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i - 2, &ref_a));
    EXPECT_TRUE(test_a_isa(ref_a));
    auto* a = test_a_deref(ref_a);
    EXPECT_EQ(i, a->data());
    iree_vm_ref_release(&ref_a);
  }

  iree_vm_list_release(list);
}

// TODO(benvanik): test primitive variant get/set.

// TODO(benvanik): test ref variant get/set.

}  // namespace
