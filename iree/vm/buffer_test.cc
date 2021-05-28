// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/builtin_types.h"
#include "iree/vm/ref_cc.h"

namespace {

class VMBufferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_vm_register_builtin_types());
  }
};

// Tests that the data allocator is correctly called when using stack
// initialization of a buffer.
TEST_F(VMBufferTest, Initialize) {
  bool did_free = false;
  iree_allocator_t test_allocator = {
      /*.self=*/&did_free,
      /*.alloc=*/NULL,
      /*.free=*/
      +[](void* self, void* ptr) { *(bool*)self = true; },
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
