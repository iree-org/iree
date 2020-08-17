// Copyright 2020 Google LLC
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

#include "iree/vm/list.h"

#include "iree/base/api.h"
#include "iree/base/ref_ptr.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/builtin_types.h"

class A : public iree::RefObject<A> {
 public:
  float data() const { return data_; }
  void set_data(float value) { data_ = value; }

 private:
  float data_ = 1.0f;
};
static iree_vm_ref_type_descriptor_t test_a_descriptor = {0};
IREE_VM_DECLARE_TYPE_ADAPTERS(test_a, A);
IREE_VM_DEFINE_TYPE_ADAPTERS(test_a, A);

class B : public iree::RefObject<B> {
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

template <typename T>
static void RegisterRefType(iree_vm_ref_type_descriptor_t* descriptor) {
  if (descriptor->type == IREE_VM_REF_TYPE_NULL) {
    descriptor->type_name = iree_make_cstring_view(typeid(T).name());
    descriptor->offsetof_counter = T::offsetof_counter();
    descriptor->destroy = T::DirectDestroy;
    IREE_CHECK_OK(iree_vm_ref_register_type(descriptor));
  }
}

static void RegisterRefTypes() {
  RegisterRefType<A>(&test_a_descriptor);
  RegisterRefType<B>(&test_b_descriptor);
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
    iree_vm_value_t value = iree_vm_value_make_i32(i);
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
    iree_vm_ref_t ref_a = MakeRef<A>(i);
    IREE_ASSERT_OK(iree_vm_list_set_ref_move(list, i, &ref_a));
  }

  for (iree_host_size_t i = 0; i < 5; ++i) {
    iree_vm_ref_t ref_a{0};
    IREE_ASSERT_OK(iree_vm_list_get_ref_retain(list, i, &ref_a));
    EXPECT_TRUE(test_a_isa(&ref_a));
    auto* a = test_a_deref(&ref_a);
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
    iree_vm_value_t value = iree_vm_value_make_i32(i);
    IREE_ASSERT_OK(iree_vm_list_set_value(list, i, &value));
  }
  for (iree_host_size_t i = 5; i < 10; ++i) {
    iree_vm_ref_t ref_a = MakeRef<A>(i);
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
    EXPECT_TRUE(test_a_isa(&ref_a));
    auto* a = test_a_deref(&ref_a);
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

// TODO(benvanik): test resize value.

// TODO(benvanik): test resize ref.

// TODO(benvanik): test resize variant.

// TODO(benvanik): test value get/set.

// TODO(benvanik): test value conversion.

// TODO(benvanik): test ref get/set.

// TODO(benvanik): test primitive variant get/set.

// TODO(benvanik): test ref variant get/set.

}  // namespace
