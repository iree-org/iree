// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Contains the test definitions applied to all loop implementations:
#include "iree/base/loop_test.h"

void AllocateLoop(iree_status_t* out_status, iree_allocator_t allocator,
                  iree_loop_t* out_loop) {
  *out_loop = iree_loop_inline(out_status);
}

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop) {}

// Tests usage of external storage for the inline ringbuffer.
// The standard tests all use loop allocated stack storage while this one uses
// the storage we control. Real applications could put that storage in .rwdata
// somewhere or alias it with other storage (arenas/etc).
TEST(LoopInlineTest, ExternalStorage) {
  IREE_TRACE_SCOPE();

  iree_loop_inline_storage_t storage = {{0xCD}, iree_ok_status()};
  auto loop = iree_loop_inline_initialize(&storage);

  // Issue a call that adds 1 to a counter until it reaches kCountUpTo.
  static const int kCountUpTo = 128;
  struct user_data_t {
    int counter = 0;
  } user_data;
  static const iree_loop_callback_fn_t callback_fn =
      +[](void* user_data_ptr, iree_loop_t loop, iree_status_t status) {
        auto* user_data = reinterpret_cast<user_data_t*>(user_data_ptr);
        if (++user_data->counter < kCountUpTo) {
          return iree_loop_call(loop, IREE_LOOP_PRIORITY_DEFAULT, callback_fn,
                                user_data);
        }
        return iree_ok_status();
      };
  IREE_ASSERT_OK(iree_loop_call(loop, IREE_LOOP_PRIORITY_DEFAULT, callback_fn,
                                &user_data));
  EXPECT_EQ(user_data.counter, kCountUpTo);
  IREE_ASSERT_OK(storage.status);

  iree_loop_inline_deinitialize(&storage);
}
