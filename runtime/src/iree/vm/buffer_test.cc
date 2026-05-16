// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"

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

typedef struct test_module_t {
  // VM module interface used for retain/release testing.
  iree_vm_module_t interface;
  // Number of times the module destroy callback ran.
  int destroy_count;
} test_module_t;

static void test_module_destroy(void* self) {
  test_module_t* module = (test_module_t*)self;
  ++module->destroy_count;
}

static void test_module_initialize(test_module_t* out_module) {
  memset(out_module, 0, sizeof(*out_module));
  IREE_CHECK_OK(iree_vm_module_initialize(&out_module->interface, out_module));
  out_module->interface.destroy = test_module_destroy;
}

TEST_F(VMBufferTest, ModuleStorageBufferRetainsStorageModule) {
  test_module_t module;
  test_module_initialize(&module);

  uint8_t data[] = {0, 1, 2, 3};
  iree_vm_buffer_t buffer;
  iree_vm_buffer_initialize_module_storage(
      iree_make_byte_span(data, sizeof(data)), &module.interface, &buffer);

  iree_vm_ref_t ref0 = iree_vm_buffer_retain_ref(&buffer);
  iree_vm_ref_t ref1 = iree_vm_buffer_retain_ref(&buffer);

  iree_vm_module_release(&module.interface);
  EXPECT_EQ(0, module.destroy_count);

  iree_vm_ref_release(&ref0);
  EXPECT_EQ(0, module.destroy_count);

  iree_vm_ref_release(&ref1);
  EXPECT_EQ(1, module.destroy_count);

  iree_vm_buffer_deinitialize(&buffer);
}

}  // namespace
