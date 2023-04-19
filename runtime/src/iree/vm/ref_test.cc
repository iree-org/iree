// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/ref.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/instance.h"
#include "iree/vm/ref.h"

namespace {

using InstancePtr =
    std::unique_ptr<iree_vm_instance_t, decltype(&iree_vm_instance_release)>;
static InstancePtr MakeInstance() {
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));
  return InstancePtr(instance, iree_vm_instance_release);
}

class A : public iree::vm::RefObject<A> {
 public:
  static iree_vm_ref_type_t kTypeID;

  int data() const { return data_; }

 private:
  int data_ = 1;
};
iree_vm_ref_type_t A::kTypeID = IREE_VM_REF_TYPE_NULL;

class B : public iree::vm::RefObject<B> {
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
static iree_vm_ref_t MakeRef(InstancePtr& instance, const char* type_name) {
  // Safe to do multiple times, so we do it to ensure the tests don't care what
  // order they run in/don't need to preregister types.
  static iree_vm_ref_type_descriptor_t descriptor = {0};
  static iree_vm_ref_type_t registration = 0;
  descriptor.type_name = iree_make_cstring_view(type_name);
  descriptor.offsetof_counter = T::offsetof_counter();
  descriptor.destroy = T::DirectDestroy;
  IREE_CHECK_OK(iree_vm_instance_register_type(instance.get(), &descriptor,
                                               &registration));
  T::kTypeID = registration;

  iree_vm_ref_t ref = {0};
  IREE_CHECK_OK(iree_vm_ref_wrap_assign(new T(), T::kTypeID, &ref));
  return ref;
}

// WARNING: this is an implementation detail and must never be relied on - it's
// only here to test the expected behavior.
static int32_t ReadCounter(iree_vm_ref_t* ref) {
  return iree_atomic_load_int32((iree_atomic_ref_count_t*)ref->ptr +
                                    (ref->type & IREE_VM_REF_TYPE_TAG_BIT_MASK),
                                iree_memory_order_seq_cst);
}

}  // namespace

IREE_VM_DECLARE_TYPE_ADAPTERS(ref_object_c, ref_object_c_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(ref_object_c, ref_object_c_t);

namespace {

static void RegisterTypeC(InstancePtr& instance) {
  static iree_vm_ref_type_descriptor_t descriptor = {0};
  descriptor.type_name = iree_make_cstring_view("CType");
  descriptor.offsetof_counter = offsetof(ref_object_c_t, ref_object.counter) /
                                IREE_VM_REF_COUNTER_ALIGNMENT;
  descriptor.destroy =
      +[](void* ptr) { delete reinterpret_cast<ref_object_c_t*>(ptr); };
  IREE_CHECK_OK(iree_vm_instance_register_type(instance.get(), &descriptor,
                                               &ref_object_c_registration));
}

// Tests type registration and lookup.
TEST(VMRefTest, TypeRegistration) {
  auto instance = MakeInstance();
  RegisterTypeC(instance);
  ASSERT_NE(0, iree_vm_instance_lookup_type(instance.get(),
                                            iree_make_cstring_view("CType")));
  ASSERT_EQ(
      0, iree_vm_instance_lookup_type(
             instance.get(), iree_make_cstring_view("asodjfaoisdjfaoisdfj")));
}

// Tests wrapping a simple C struct.
TEST(VMRefTest, WrappingCStruct) {
  auto instance = MakeInstance();
  RegisterTypeC(instance);
  iree_vm_ref_t ref = {0};
  IREE_EXPECT_OK(iree_vm_ref_wrap_assign(new ref_object_c_t(),
                                         ref_object_c_registration, &ref));
  EXPECT_EQ(1, ReadCounter(&ref));
  iree_vm_ref_release(&ref);
}

// Tests wrapping a C++ RefObject with a vtable.
TEST(VMRefTest, WrappingSubclassedRefObject) {
  struct BaseType : public iree::vm::RefObject<BaseType> {
    virtual ~BaseType() = default;
    virtual int DoSomething() = 0;
  };
  static int allocated_derived_types = 0;
  struct DerivedType : public BaseType {
    DerivedType() { ++allocated_derived_types; }
    ~DerivedType() override { --allocated_derived_types; }
    int DoSomething() override { return 123 + allocated_derived_types; }
  };

  auto instance = MakeInstance();

  static iree_vm_ref_type_descriptor_t descriptor = {0};
  static iree_vm_ref_type_t registration = 0;
  descriptor.type_name = iree_make_cstring_view("BaseType");
  descriptor.offsetof_counter = BaseType::offsetof_counter();
  descriptor.destroy = BaseType::DirectDestroy;
  IREE_ASSERT_OK(iree_vm_instance_register_type(instance.get(), &descriptor,
                                                &registration));

  allocated_derived_types = 0;

  iree_vm_ref_t ref = {0};
  IREE_EXPECT_OK(
      iree_vm_ref_wrap_assign(new DerivedType(), registration, &ref));
  EXPECT_EQ(1, ReadCounter(&ref));
  EXPECT_EQ(1, allocated_derived_types);

  EXPECT_EQ(123 + 1, reinterpret_cast<BaseType*>(ref.ptr)->DoSomething());

  iree_vm_ref_release(&ref);
  EXPECT_EQ(0, allocated_derived_types);
}

// Tests that wrapping releases any existing ref in out_ref.
TEST(VMRefTest, WrappingReleasesExisting) {
  auto instance = MakeInstance();
  RegisterTypeC(instance);
  iree_vm_ref_t ref = {0};
  iree_vm_ref_wrap_assign(new ref_object_c_t(), ref_object_c_registration,
                          &ref);
  EXPECT_EQ(1, ReadCounter(&ref));
  iree_vm_ref_release(&ref);
}

// Checking null refs is fine.
TEST(VMRefTest, CheckNull) {
  iree_vm_ref_t null_ref = {0};
  IREE_EXPECT_OK(iree_vm_ref_check(null_ref, IREE_VM_REF_TYPE_NULL));
  iree_status_t status =
      iree_vm_ref_check(null_ref, static_cast<iree_vm_ref_type_t>(1234));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

// Tests type checks.
TEST(VMRefTest, Check) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  IREE_EXPECT_OK(iree_vm_ref_check(a_ref, A::kTypeID));
  iree_status_t status = iree_vm_ref_check(a_ref, B::kTypeID);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
  iree_vm_ref_release(&a_ref);
}

// Tests retaining a null ref does nothing.
TEST(VMRefTest, RetainNull) {
  iree_vm_ref_t null_ref_0 = {0};
  iree_vm_ref_t null_ref_1 = {0};
  iree_vm_ref_retain(&null_ref_0, &null_ref_1);
}

// Tests that retaining into itself is a no-op.
TEST(VMRefTest, RetainIntoSelf) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_retain(&a_ref, &a_ref);
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests that retaining into out_ref releases the existing contents.
TEST(VMRefTest, RetainReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t b_ref = MakeRef<B>(instance, "BType");
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
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
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
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref_0, &a_ref_1);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(2, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
  iree_vm_ref_release(&a_ref_1);
}

// Tests that is_move=true does not increment the ref count.
TEST(VMRefTest, RetainOrMoveMoving) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_retain_or_move(/*is_move=*/1, &a_ref_0, &a_ref_1);
  IREE_EXPECT_OK(iree_vm_ref_check(a_ref_0, IREE_VM_REF_TYPE_NULL));
  iree_vm_ref_release(&a_ref_1);
}

// Tests that retaining into itself just increments the ref count.
TEST(VMRefTest, RetainOrMoveRetainingIntoSelf) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref, &a_ref);
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests that moving into itself is a no-op.
TEST(VMRefTest, RetainOrMoveMovingIntoSelf) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_retain_or_move(/*is_move=*/1, &a_ref, &a_ref);
  IREE_EXPECT_OK(iree_vm_ref_check(a_ref, A::kTypeID));
  iree_vm_ref_release(&a_ref);
}

// Tests that retaining into out_ref releases the existing contents.
TEST(VMRefTest, RetainOrMoveRetainingReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t b_ref = MakeRef<B>(instance, "BType");
  iree_vm_ref_retain_or_move(/*is_move=*/0, &a_ref, &b_ref);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(2, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
  iree_vm_ref_release(&b_ref);
}

// Tests that moving into out_ref releases the existing contents.
TEST(VMRefTest, RetainOrMoveMovingReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t b_ref = MakeRef<B>(instance, "BType");
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
  auto instance = MakeInstance();

  // Retain.
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &a_ref_0, A::kTypeID, &a_ref_1));
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(2, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
  iree_vm_ref_release(&a_ref_1);

  // Move.
  iree_vm_ref_t b_ref_0 = MakeRef<B>(instance, "BType");
  iree_vm_ref_t b_ref_1 = {0};
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/1, &b_ref_0, B::kTypeID, &b_ref_1));
  EXPECT_EQ(0, iree_vm_ref_equal(&b_ref_0, &b_ref_1));
  EXPECT_EQ(1, ReadCounter(&b_ref_1));
  iree_vm_ref_release(&b_ref_1);
}

// Tests that types are verified and retains/moves fail if types don't match.
TEST(VMRefTest, RetainOrMoveCheckedMismatch) {
  auto instance = MakeInstance();

  // Retain.
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  iree_status_t status = iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &a_ref_0, B::kTypeID, &a_ref_1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(1, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);

  // Move.
  iree_vm_ref_t b_ref_0 = MakeRef<B>(instance, "BType");
  iree_vm_ref_t b_ref_1 = {0};
  status = iree_vm_ref_retain_or_move_checked(
      /*is_move=*/1, &b_ref_0, A::kTypeID, &b_ref_1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
  EXPECT_EQ(1, ReadCounter(&b_ref_0));
  iree_vm_ref_release(&b_ref_0);
}

// Tests that existing references are released when being overwritten.
TEST(VMRefTest, RetainOrMoveCheckedReleasesExistingNull) {
  auto instance = MakeInstance();
  iree_vm_ref_t null_ref = {0};
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  IREE_EXPECT_OK(iree_vm_ref_retain_or_move_checked(
      /*is_move=*/0, &null_ref, A::kTypeID, &a_ref));
}

// Tests that existing references are released when being overwritten.
TEST(VMRefTest, RetainOrMoveCheckedReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = MakeRef<A>(instance, "AType");
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
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_assign(&a_ref_0, &a_ref_1);
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref_0, &a_ref_1));
  EXPECT_EQ(1, ReadCounter(&a_ref_0));
  iree_vm_ref_release(&a_ref_0);
}

// Tests that assigning into itself is a no-op.
TEST(VMRefTest, AssignSelf) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_assign(&a_ref, &a_ref);
  EXPECT_EQ(1, ReadCounter(&a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests that assigning into out_ref releases the existing contents.
TEST(VMRefTest, AssignReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t b_ref = MakeRef<B>(instance, "BType");
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
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = {0};
  iree_vm_ref_move(&a_ref_0, &a_ref_1);
  IREE_EXPECT_OK(iree_vm_ref_check(a_ref_0, IREE_VM_REF_TYPE_NULL));
  iree_vm_ref_release(&a_ref_1);
}

// Tests that moving into itself is a no-op.
TEST(VMRefTest, MovingIntoSelf) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_move(&a_ref, &a_ref);
  IREE_EXPECT_OK(iree_vm_ref_check(a_ref, A::kTypeID));
  iree_vm_ref_release(&a_ref);
}

// Tests that moving into out_ref releases the existing contents.
TEST(VMRefTest, MovingReleasesExisting) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref_0 = MakeRef<A>(instance, "AType");
  iree_vm_ref_t a_ref_1 = MakeRef<A>(instance, "AType");
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
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t null_ref = {0};
  EXPECT_EQ(1, iree_vm_ref_equal(&a_ref, &a_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref, &null_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&null_ref, &a_ref));
  iree_vm_ref_release(&a_ref);
}

// Tests comparing between different types.
TEST(VMRefTest, EqualityDifferentTypes) {
  auto instance = MakeInstance();
  iree_vm_ref_t a_ref = MakeRef<A>(instance, "AType");
  iree_vm_ref_t b_ref = MakeRef<B>(instance, "BType");
  EXPECT_EQ(0, iree_vm_ref_equal(&a_ref, &b_ref));
  EXPECT_EQ(0, iree_vm_ref_equal(&b_ref, &a_ref));
  iree_vm_ref_release(&b_ref);
  iree_vm_ref_release(&a_ref);
}

// Tests that in-place assignment of vm::ref when used with C create functions
// properly tracks both the pointer and the iree_vm_ref_type_t.
TEST(VMRefTest, InPlaceAssignment) {
  auto instance = MakeInstance();
  RegisterTypeC(instance);
  auto create = [&](ref_object_c_t** out_object) {
    *out_object = new ref_object_c_t();
  };
  iree::vm::ref<ref_object_c_t> ref;
  EXPECT_FALSE(ref);
  EXPECT_EQ(nullptr, ref.get());
  EXPECT_EQ(IREE_VM_REF_TYPE_NULL, ref.type());
  create(&ref);
  EXPECT_TRUE(ref);
  EXPECT_NE(nullptr, ref.get());
  EXPECT_EQ(ref_object_c_type(), ref.type());
  ref.reset();
  EXPECT_FALSE(ref);
  EXPECT_EQ(nullptr, ref.get());
  EXPECT_EQ(IREE_VM_REF_TYPE_NULL, ref.type());
}

}  // namespace

struct ref_object_d_t {
  iree_vm_ref_object_t ref_object = {1};
  int data = 1;
};

IREE_VM_DECLARE_TYPE_ADAPTERS(ref_object_d, ref_object_d_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(ref_object_d, ref_object_d_t);

namespace {

// Tests that C++ ref<T> instances can be initialized even if the type is not
// yet registered. This happens if whatever C++ object/scope owns the instance
// has fields that are initialized prior to the instance/module loading.
TEST(VMRefTest, UnregisteredType) {
  iree::vm::ref<ref_object_d_t> null_ref;
  EXPECT_FALSE(null_ref);
  EXPECT_EQ(nullptr, null_ref.get());
  EXPECT_EQ(nullptr, null_ref.release());
  EXPECT_EQ(IREE_VM_REF_TYPE_NULL, null_ref.type());
  null_ref.reset();  // don't die
  auto retained_ref = iree::vm::retain_ref<ref_object_d_t>(nullptr);
  EXPECT_FALSE(retained_ref);
  auto assigned_ref = iree::vm::assign_ref<ref_object_d_t>(nullptr);
  EXPECT_FALSE(assigned_ref);
}

}  // namespace
