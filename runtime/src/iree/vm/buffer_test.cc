// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/vm/instance.h"

namespace {

static iree_vm_instance_t* instance = NULL;
struct VMBufferTest : public ::testing::Test {
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance));
  }
  static void TearDownTestSuite() { iree_vm_instance_release(instance); }
};

// Tests that the data allocator is correctly called when using stack
// initialization of a buffer.
TEST_F(VMBufferTest, Initialize) {
  bool did_free = false;
  iree_allocator_t test_allocator = {
      /*.self=*/&did_free,
      /*.ctl=*/
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** inout_ptr) {
        if (command == IREE_ALLOCATOR_COMMAND_FREE) {
          *(bool*)self = true;
        }
        return iree_ok_status();
      },
  };

  uint32_t data[] = {0, 1, 2, 3};
  iree_vm_buffer_t buffer;
  iree_vm_buffer_initialize(
      IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      iree_make_byte_span(data, sizeof(data)), test_allocator, &buffer);

  ASSERT_FALSE(did_free);
  iree_vm_buffer_deinitialize(&buffer);
  ASSERT_TRUE(did_free);
}

}  // namespace
