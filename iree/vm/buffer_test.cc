// Copyright 2021 Google LLC
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
