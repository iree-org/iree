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

#include "iree/vm/ref.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/ref_ptr.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class A : public iree::RefObject<A> {
 public:
  static iree_vm_ref_type_t kTypeID;

  int data() const { return data_; }

 private:
  int data_ = 1;
};
iree_vm_ref_type_t A::kTypeID = IREE_VM_REF_TYPE_NULL;

class B : public iree::RefObject<B> {
 public:
  static iree_vm_ref_type_t kTypeID;

  int data() const { return data_; }

 private:
  int data_ = 2;
};
iree_vm_ref_type_t B::kTypeID = IREE_VM_REF_TYPE_NULL;

struct ref_object_c_t {
  iree_vm_ref_object_t ref_object = {1};
  int data = 1;
};

template <typename T>
static iree_vm_ref_t MakeRef() {
  // Safe to do multiple times, so we do it to ensure the tests don't care what
  // order they run in/don't need to preregister types.
  static iree_vm_ref_type_descriptor_t descriptor = {0};
  if (descriptor.type == IREE_VM_REF_TYPE_NULL) {
    descriptor.type_name = iree_make_cstring_view(typeid(T).name());
    descriptor.offsetof_counter = T::offsetof_counter();
    descriptor.destroy = T::DirectDestroy;
    IREE_CHECK_OK(iree_vm_ref_register_type(&descriptor));
    T::kTypeID = descriptor.type;
  }

  iree_vm_ref_t ref = {0};
  IREE_CHECK_OK(iree_vm_ref_wrap_assign(new T(), T::kTypeID, &ref));
  return ref;
}

static intptr_t ReadCounter(iree_vm_ref_t* ref) {
  return *((intptr_t*)(((uintptr_t)ref->ptr) + ref->offsetof_counter));
}

static iree_vm_ref_type_t kCTypeID = IREE_VM_REF_TYPE_NULL;
static void RegisterTypeC() {
  static iree_vm_ref_type_descriptor_t descriptor = {0};
  if (descriptor.type == IREE_VM_REF_TYPE_NULL) {
    descriptor.type_name =
        iree_make_cstring_view(typeid(ref_object_c_t).name());
    descriptor.offsetof_counter = offsetof(ref_object_c_t, ref_object.counter);
    descriptor.destroy =
        +[](void* ptr) { delete reinterpret_cast<ref_object_c_t*>(ptr); };
    IREE_CHECK_OK(iree_vm_ref_register_type(&descriptor));
    kCTypeID = descriptor.type;
  }
}

// Tests type registration and lookup.
TEST(VMRefTest, TypeRegistration) {
  RegisterTypeC();
  ASSERT_NE(nullptr, iree_vm_ref_lookup_registered_type(iree_make_cstring_view(
                         typeid(ref_object_c_t).name())));
  ASSERT_EQ(nullptr, iree_vm_ref_lookup_registered_type(
                         iree_make_cstring_view("asodjfaoisdjfaoisdfj")));
}

// Tests wrapping a simple C struct.
TEST(VMRefTest, WrappingCStruct) {
  RegisterTypeC();
  iree_vm_ref_t ref = {0};
  IREE_EXPECT_OK(iree_vm_ref_wrap_assign(new ref_object_c_t(), kCTypeID, &ref));
  EXPECT_EQ(1, ReadCounter(&ref));
  iree_vm_ref_release(&ref);
}

// Tests wrapping a C++ RefObject with a vtable.
TEST(VMRefTest, WrappingSubclassedRefObject) {
  struct BaseType : public iree::RefObject<BaseType> {
    virtual ~BaseType() = default;
    virtual int DoSomething() = 0;
  };
  static int allocated_derived_types = 0;
  struct DerivedType : public BaseType {
    DerivedType() { ++allocated_derived_types; }
    ~DerivedType() override { --allocated_derived_types; }
    int DoSomething() override { return 123 + allocated_derived_types; }
  };

  static iree_vm_ref_type_descriptor_t descriptor;
  descriptor.type_name = iree_make_cstring_view(typeid(BaseType).name());
  descriptor.offsetof_counter = BaseType::offsetof_counter();
  descriptor.destroy = BaseType::DirectDestroy;
  IREE_ASSERT_OK(iree_vm_ref_register_type(&descriptor));

  allocated_derived_types = 0;

  iree_vm_ref_t ref = {0};
  IREE_EXPECT_OK(
      iree_vm_ref_wrap_assign(new DerivedType(), descriptor.type, &ref));
  EXPECT_EQ(1, ReadCounter(&ref));
  EXPECT_EQ(1, allocated_derived_types);

  EXPECT_EQ(123 + 1, reinterpret_cast<BaseType*>(ref.ptr)->DoSomething());

  iree_vm_ref_release(&ref);
  EXPECT_EQ(0, allocated_derived_types);
}

// Tests that wrapping a type that has not been registered fails.
TEST(VMRefTest, WrappingRequriesTypeRegistration) {
  iree_vm_ref_t ref = {0};
  int dummy = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ::iree::Status(iree_vm_ref_wrap_assign(
          &dummy, static_cast<iree_vm_ref_type_t>(1234), &ref)));
}

// Tests that wrapping releases any existing ref in out_ref.
TEST(VMRefTest, WrappingReleasesExisting) {
  RegisterTypeC();
  iree_vm_ref_t ref = {0};
  iree_vm_ref_wrap_assign(new ref_object_c_t(), kCTypeID, &ref);
  EXPECT_EQ(1, ReadCounter(&ref));
  iree_vm_ref_release(&ref);
}

// Checking null refs is fine.
TEST(VMRefTest, CheckNull) {
  iree_vm_ref_t null_ref = {0};
  IREE_EXPECT_OK(iree_vm_ref_check(&null_ref, IREE_VM_REF_TYPE_NULL));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ::iree::Status(iree_vm_ref_check(
                            &null_ref, static_cast<iree_vm_ref_type_t>(1234))));
}

// Tests type checks.
TEST(VMRefTest, Check) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  IREE_EXPECT_OK(iree_vm_ref_check(&a_ref, A::kTypeID));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ::iree::Status(iree_vm_ref_check(&a_ref, B::kTypeID)));
  iree_vm_ref_release(&a_ref);
}

// Tests retaining a null ref does nothing.
TEST(VMRefTest, RetainNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  iree_vm_ref_retain(&null_ref_0, &null_ref_1);
}

// Tests that retaining into itself only increments the count.
TEST(VMRefTest, RetainIntoSelf) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_retain(&a_ref, &a_ref);
  EXPECT_EQ(2, ReadCounter(&a_ref));

  iree_vm_ref_t last_ref = {0};
  iree_vm_ref_assign(&a_ref, &last_ref);
  iree_vm_ref_release(&a_ref);
  iree_vm_ref_release(&last_ref);
}

// Tests that retaining into out_ref releases the existing contents.
TEST(VMRefTest, RetainReleasesExisting) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t b_ref = MakeRef<B>();
  iree_vm_ref_retain(&a_ref, &b_ref);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(2, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
  iree_vm_ref_release(&b_ref);
}

// Tests that null refs are always fine.
TEST(VMRefTest, RetainCheckedNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  IREE_EXPECT_OK(
      iree_vm_ref_retain_checked(&null_ref_0, A::kTypeID, &null_ref_1));
}

// Tests that types are verified and retains fail if types don't match.
TEST(VMRefTest, RetainChecked) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_checked(&a_ref_0, A::kTypeID, &a_ref_1));
  iree_vm_ref_release(&a_ref_0);
  iree_vm_ref_release(&a_ref_1);
}

// Tests that working with null refs is fine.
TEST(VMRefTest, RetainOrMoveNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  iree_vm_ref_retain_or_move(/*is_move=*/0, &null_ref_0, &null_ref_1);
  iree_vm_ref_retain_or_move(/*is_move=*/1, &null_ref_0, &null_ref_1);
}

// Tests that is_move=false increments the ref count.
TEST(VMRefTest, RetainOrMoveRetaining) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref_0, &a_ref_1);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(2, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
  iree_vm_ref_release(&a_ref_1);
}

// Tests that is_move=true does not increment the ref count.
TEST(VMRefTest, RetainOrMoveMoving) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_retain_or_move(/*is_move=*/1, &a_ref_0, &a_ref_1);
  IREE_EXPECT_OK(iree_vm_ref_check(&a_ref_0, IREE_VM_REF_TYPE_NULL));
  iree_vm_ref_release(&a_ref_1);
}

// Tests that retaining into itself just increments the ref count.
TEST(VMRefTest, RetainOrMoveRetainingIntoSelf) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref, &a_ref);
  EXPECT_EQ(2, ReadCounter(&a_ref));

  iree_vm_ref_t last_ref = {0};
  iree_vm_ref_assign(&a_ref, &last_ref);
  iree_vm_ref_release(&a_ref);
  iree_vm_ref_release(&last_ref);
}

// Tests that moving into itself is a no-op.
TEST(VMRefTest, RetainOrMoveMovingIntoSelf) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_retain_or_move(/*is_move=*/1, &a_ref, &a_ref);
  IREE_EXPECT_OK(iree_vm_ref_check(&a_ref, A::kTypeID));
  iree_vm_ref_release(&a_ref);
}

// Tests that retaining into out_ref releases the existing contents.
TEST(VMRefTest, RetainOrMoveRetainingReleasesExisting) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t b_ref = MakeRef<B>();
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref, &b_ref);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(2, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
  iree_vm_ref_release(&b_ref);
}

// Tests that moving into out_ref releases the existing contents.
TEST(VMRefTest, RetainOrMoveMovingReleasesExisting) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t b_ref = MakeRef<B>();
  iree_vm_ref_retain_or_move(/*is_move=*/1, &a_ref, &b_ref);
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(1, ReadCounter(&b_ref));
  iree_vm_ref_release(&b_ref);
}

// Tests that null refs are always fine.
TEST(VMRefTest, RetainOrMoveCheckedNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &null_ref_0, A::kTypeID, &null_ref_1));
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/1, &null_ref_0, A::kTypeID, &null_ref_1));
}

// Tests that retains/moves work when types match.
TEST(VMRefTest, RetainOrMoveCheckedMatch) {
  // Retain.
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &a_ref_0, A::kTypeID, &a_ref_1));
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(2, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
  iree_vm_ref_release(&a_ref_1);

  // Move.
  iree_vm_ref_t b_ref_0 = MakeRef<B>();
  iree_vm_ref_t b_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/1, &b_ref_0, B::kTypeID, &b_ref_1));
  EXPECT_EQ(0, iree_vm_ref_equal(&b_ref_0, &b_ref_1));
  EXPECT_EQ(1, ReadCounter(&b_ref_1));
  iree_vm_ref_release(&b_ref_1);
}

// Tests that types are verified and retains/moves fail if types don't match.
TEST(VMRefTest, RetainOrMoveCheckedMismatch) {
  // Retain.
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ::iree::Status(iree_vm_ref_retain_or_move_checked(
                            /*is_move=*/0, &a_ref_0, B::kTypeID, &a_ref_1)));
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(1, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);

  // Move.
  iree_vm_ref_t b_ref_0 = MakeRef<B>();
  iree_vm_ref_t b_ref_1 = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ::iree::Status(iree_vm_ref_retain_or_move_checked(
                            /*is_move=*/1, &b_ref_0, A::kTypeID, &b_ref_1)));
  EXPECT_EQ(1, ReadCounter(&b_ref_0));
  iree_vm_ref_release(&b_ref_0);
}

// Tests that existing references are released when being overwritten.
TEST(VMRefTest, RetainOrMoveCheckedReleasesExistingNull) {
  iree_vm_ref_t null_ref = {0};
  iree_vm_ref_t a_ref = MakeRef<A>();
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &null_ref, A::kTypeID, &a_ref));
}

// Tests that existing references are released when being overwritten.
TEST(VMRefTest, RetainOrMoveCheckedReleasesExisting) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = MakeRef<A>();
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/1, &a_ref_0, A::kTypeID, &a_ref_1));
  iree_vm_ref_release(&a_ref_1);
}

// Checks that assigning null refs is fine.
TEST(VMRefTest, AssignNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  iree_vm_ref_assign(&null_ref_0, &null_ref_1);
}

// Tests that assigning does not reset the source ref nor inc the ref count.
TEST(VMRefTest, Assign) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_assign(&a_ref_0, &a_ref_1);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(1, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
}

// Tests that assigning into itself is a no-op.
TEST(VMRefTest, AssignSelf) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_assign(&a_ref, &a_ref);
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests that assigning into out_ref releases the existing contents.
TEST(VMRefTest, AssignReleasesExisting) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t b_ref = MakeRef<B>();
  iree_vm_ref_assign(&a_ref, &b_ref);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
  // NOTE: do not release b - it was just assigned!
}

// Checks that moving null refs is fine.
TEST(VMRefTest, MovingNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  iree_vm_ref_move(&null_ref_0, &null_ref_1);
}

// Tests that moving resets the source ref.
TEST(VMRefTest, MovingResetsSource) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_move(&a_ref_0, &a_ref_1);
  IREE_EXPECT_OK(iree_vm_ref_check(&a_ref_0, IREE_VM_REF_TYPE_NULL));
  iree_vm_ref_release(&a_ref_1);
}

// Tests that moving into itself is a no-op.
TEST(VMRefTest, MovingIntoSelf) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_move(&a_ref, &a_ref);
  IREE_EXPECT_OK(iree_vm_ref_check(&a_ref, A::kTypeID));
  iree_vm_ref_release(&a_ref);
}

// Tests that moving into out_ref releases the existing contents.
TEST(VMRefTest, MovingReleasesExisting) {
  iree_vm_ref_t a_ref_0 = MakeRef<A>();
  iree_vm_ref_t a_ref_1 = MakeRef<A>();
  iree_vm_ref_move(&a_ref_0, &a_ref_1);
  iree_vm_ref_release(&a_ref_1);
}

// Null references should always be equal.
TEST(VMRefTest, EqualityNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  EXPECT_EQ(1, iree_vm_ref_equal(&null_ref_0, &null_ref_0));
  EXPECT_EQ(1, iree_vm_ref_equal(&null_ref_0, &null_ref_1));
  EXPECT_EQ(1, iree_vm_ref_equal(&null_ref_1, &null_ref_0));
}

// Tests comparing with self and against null.
TEST(VMRefTest, EqualitySelfOrNull) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t null_ref = {0};
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &a_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref, &null_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&null_ref, &a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests comparing between different types.
TEST(VMRefTest, EqualityDifferentTypes) {
  iree_vm_ref_t a_ref = MakeRef<A>();
  iree_vm_ref_t b_ref = MakeRef<B>();
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&b_ref, &a_ref));
  iree_vm_ref_release(&b_ref);
  iree_vm_ref_release(&a_ref);
}

}  // namespace
