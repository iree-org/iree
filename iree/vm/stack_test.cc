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

#include "iree/vm/stack.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/base/ref_ptr.h"
#include "iree/testing/gtest.h"
#include "iree/vm/ref.h"

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
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_a, NULL, NULL, &frame_a));
  EXPECT_EQ(0, frame_a->function.ordinal);
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  iree_vm_function_t function_b = {MODULE_B_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 1};
  iree_vm_stack_frame_t* frame_b = nullptr;
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_b, NULL, NULL, &frame_b));
  EXPECT_EQ(1, frame_b->function.ordinal);
  EXPECT_EQ(frame_b, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(frame_a, iree_vm_stack_parent_frame(stack));

  IREE_EXPECT_OK(iree_vm_stack_function_leave(stack, NULL, NULL));
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));
  IREE_EXPECT_OK(iree_vm_stack_function_leave(stack, NULL, NULL));
  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  iree_vm_stack_deinitialize(stack);
}

// Tests stack cleanup with unpopped frames (like during failure teardown).
TEST(VMStackTest, DeinitWithRemainingFrames) {
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_a, NULL, NULL, &frame_a));
  EXPECT_EQ(0, frame_a->function.ordinal);
  EXPECT_EQ(frame_a, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  // Don't pop the last frame before deinit; it should handle it.
  iree_vm_stack_deinitialize(stack);
}

// Tests stack overflow detection.
TEST(VMStackTest, StackOverflow) {
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  // Fill the entire stack up to the max.
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  bool did_overflow = false;
  for (int i = 0; i < 99999; ++i) {
    iree_vm_stack_frame_t* frame_a = nullptr;
    iree_status_t status =
        iree_vm_stack_function_enter(stack, function_a, NULL, NULL, &frame_a);
    if (iree_status_is_resource_exhausted(status)) {
      // Hit the stack overflow, as expected.
      did_overflow = true;
      break;
    }
    IREE_EXPECT_OK(status);
  }
  ASSERT_TRUE(did_overflow);

  iree_vm_stack_deinitialize(stack);
}

// Tests unbalanced stack popping.
TEST(VMStackTest, UnbalancedPop) {
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  EXPECT_TRUE(iree_status_is_failed_precondition(
      iree_vm_stack_function_leave(stack, NULL, NULL)));

  iree_vm_stack_deinitialize(stack);
}

// Tests module state reuse and querying.
TEST(VMStackTest, ModuleStateQueries) {
  iree_vm_state_resolver_t state_resolver = {nullptr, SentinelStateResolver};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  EXPECT_EQ(nullptr, iree_vm_stack_current_frame(stack));
  EXPECT_EQ(nullptr, iree_vm_stack_parent_frame(stack));

  module_a_state_resolve_count = 0;
  module_b_state_resolve_count = 0;

  // [A (queried)]
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_a, NULL, NULL, &frame_a));
  EXPECT_EQ(MODULE_A_STATE_SENTINEL, frame_a->module_state);
  EXPECT_EQ(1, module_a_state_resolve_count);

  // [A, B (queried)]
  iree_vm_function_t function_b = {MODULE_B_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 1};
  iree_vm_stack_frame_t* frame_b = nullptr;
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_b, NULL, NULL, &frame_b));
  EXPECT_EQ(MODULE_B_STATE_SENTINEL, frame_b->module_state);
  EXPECT_EQ(1, module_b_state_resolve_count);

  // [A, B, B (reuse)]
  IREE_EXPECT_OK(
      iree_vm_stack_function_enter(stack, function_b, NULL, NULL, &frame_b));
  EXPECT_EQ(MODULE_B_STATE_SENTINEL, frame_b->module_state);
  EXPECT_EQ(1, module_b_state_resolve_count);

  IREE_EXPECT_OK(iree_vm_stack_function_leave(stack, NULL, NULL));
  IREE_EXPECT_OK(iree_vm_stack_function_leave(stack, NULL, NULL));
  IREE_EXPECT_OK(iree_vm_stack_function_leave(stack, NULL, NULL));

  iree_vm_stack_deinitialize(stack);
}

// Tests that module state query failures propagate to callers correctly.
TEST(VMStackTest, ModuleStateQueryFailure) {
  iree_vm_state_resolver_t state_resolver = {
      nullptr,
      +[](void* state_resolver, iree_vm_module_t* module,
          iree_vm_module_state_t** out_module_state) -> iree_status_t {
        // NOTE: always failing.
        return IREE_STATUS_INTERNAL;
      }};
  IREE_VM_INLINE_STACK_INITIALIZE(stack, state_resolver, IREE_ALLOCATOR_SYSTEM);

  // Push should fail if we can't query state, status should propagate.
  iree_vm_function_t function_a = {MODULE_A_SENTINEL,
                                   IREE_VM_FUNCTION_LINKAGE_INTERNAL, 0};
  iree_vm_stack_frame_t* frame_a = nullptr;
  EXPECT_EQ(IREE_STATUS_INTERNAL, iree_vm_stack_function_enter(
                                      stack, function_a, NULL, NULL, &frame_a));

  iree_vm_stack_deinitialize(stack);
}

}  // namespace
