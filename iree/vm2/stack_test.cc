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

#include "iree/vm2/stack.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/base/ref_ptr.h"
#include "iree/testing/gtest.h"
#include "iree/vm2/ref.h"

namespace {

#define MODULE_A_SENTINEL reinterpret_cast<iree_vm_module_t*>(1)
#define MODULE_B_SENTINEL reinterpret_cast<iree_vm_module_t*>(2)
#define MODULE_A_STATE_SENTINEL reinterpret_cast<iree_vm_module_state_t*>(101)
#define MODULE_B_STATE_SENTINEL reinterpret_cast<iree_vm_module_state_t*>(102)

static int module_a_state_resolve_count = 0;
static int module_b_state_resolve_count = 0;
static iree_status_t SentinelStateResolver(
    void* state_resolver, iree_vm_module_t* module,
    iree_vm_module_state_t** out_module_state) {
  if (module == MODULE_A_SENTINEL) {
    ++module_a_state_resolve_count;
    *out_module_state = MODULE_A_STATE_SENTINEL;
    return IREE_STATUS_OK;
  } else if (module == MODULE_B_SENTINEL) {
    ++module_b_state_resolve_count;
    *out_module_state = MODULE_B_STATE_SENTINEL;
    return IREE_STATUS_OK;
  }
  return IREE_STATUS_NOT_FOUND;
}

// Tests simple stack usage, mainly just for demonstration.
TEST(VMStackTest, Usage) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));
  EXPECT_EQ(0, frame_a->function.ordinal);
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  iree_vm_function_t function_b = {MODULE_B_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 1};
  iree_vm_stack_frame_t* frame_b = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_b, &frame_b));
  EXPECT_EQ(1, frame_b->function.ordinal);
  EXPECT_EQ(frame_b, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(frame_a, iree_vm_stack_parent_frame(stack.get()));

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

// Tests stack cleanup with unpopped frames (like during failure teardown).
TEST(VMStackTest, DeinitWithRemainingFrames) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));
  EXPECT_EQ(0, frame_a->function.ordinal);
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  // Don't pop the last frame before deinit; it should handle it.
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack.get()));
}

// Tests stack overflow detection.
TEST(VMStackTest, StackOverflow) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  // Fill the entire stack up to the max.
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  for (int i = 0; i < IREE_MAX_STACK_DEPTH; ++i) {
    iree_vm_stack_frame_t* frame_a = nullptr;
    EXPECT_EQ(IREE_STATUS_OK,
              iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));
  }

  // Try to push on one more frame.
  iree_vm_function_t function_b = {MODULE_B_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 1};
  iree_vm_stack_frame_t* frame_b = nullptr;
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED,
            iree_vm_stack_function_enter(stack.get(), function_b, &frame_b));

  // Should still be frame A.
  EXPECT_EQ(0, iree_vm_stack_current_frame(stack.get())->function.ordinal);

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

// Tests unbalanced stack popping.
TEST(VMStackTest, UnbalancedPop) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  EXPECT_EQ(IREE_STATUS_FAILED_PRECONDITION,
            iree_vm_stack_function_leave(stack.get()));

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

// Tests module state reuse and querying.
TEST(VMStackTest, ModuleStateQueries) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack.get()));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack.get()));

  module_a_state_resolve_count = 0;
  module_b_state_resolve_count = 0;

  // [A (queried)]
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));
  EXPECT_EQ(MODULE_A_STATE_SENTINEL, frame_a->module_state);
  EXPECT_EQ(1, module_a_state_resolve_count);

  // [A, B (queried)]
  iree_vm_function_t function_b = {MODULE_B_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 1};
  iree_vm_stack_frame_t* frame_b = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_b, &frame_b));
  EXPECT_EQ(MODULE_B_STATE_SENTINEL, frame_b->module_state);
  EXPECT_EQ(1, module_b_state_resolve_count);

  // [A, B, B (reuse)]
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_b, &frame_b));
  EXPECT_EQ(MODULE_B_STATE_SENTINEL, frame_b->module_state);
  EXPECT_EQ(1, module_b_state_resolve_count);

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

// Tests that module state query failures propagate to callers correctly.
TEST(VMStackTest, ModuleStateQueryFailure) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {
      nullptr,
      +[](void* state_resolver, iree_vm_module_t* module,
          iree_vm_module_state_t** out_module_state) -> iree_status_t {
        // NOTE: always failing.
        return IREE_STATUS_INTERNAL;
      }};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  // Push should fail if we can't query state, status should propagate.
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_INTERNAL,
            iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

static int dummy_object_count = 0;
class DummyObject : public iree::RefObject<DummyObject> {
 public:
  static void RegisterType() {
    iree_vm_ref_type_descriptor_t descriptor;
    descriptor.type = IREE_VM_REF_TYPE_HAL_ALLOCATOR;
    descriptor.type_name = iree_string_view_t{
        typeid(DummyObject).name(), std::strlen(typeid(DummyObject).name())};
    descriptor.offsetof_counter = DummyObject::offsetof_counter();
    descriptor.destroy = DummyObject::DirectDestroy;
    iree_vm_ref_register_builtin_type(descriptor);
  }

  DummyObject() { ++dummy_object_count; }
  ~DummyObject() { --dummy_object_count; }
};

// Tests stack frame ref register cleanup.
TEST(VMStackTest, RefRegisterCleanup) {
  auto stack = std::make_unique<iree_vm_stack_t>();
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, stack.get()));

  dummy_object_count = 0;
  DummyObject::RegisterType();

  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_stack_function_enter(stack.get(), function_a, &frame_a));
  frame_a->registers.ref_register_count = 1;
  EXPECT_EQ(IREE_STATUS_OK,
            iree_vm_ref_wrap(new DummyObject(), IREE_VM_REF_TYPE_HAL_ALLOCATOR,
                             &frame_a->registers.ref[0]));
  EXPECT_EQ(1, dummy_object_count);

  // This should release the ref for us. Heap checker will yell if it doesn't.
  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_function_leave(stack.get()));
  EXPECT_EQ(0, dummy_object_count);

  EXPECT_EQ(IREE_STATUS_OK, iree_vm_stack_deinit(stack.get()));
}

}  // namespace
