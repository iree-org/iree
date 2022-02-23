// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop_sync.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Contains the test definitions applied to all loop implementations:
#include "iree/base/loop_test.h"

void AllocateLoop(iree_status_t* out_status, iree_allocator_t allocator,
                  iree_loop_t* out_loop) {
  iree_loop_sync_options_t options = {0};
  options.max_queue_depth = 128;
  options.max_wait_count = 32;

  iree_loop_sync_t* loop_sync = NULL;
  IREE_CHECK_OK(iree_loop_sync_allocate(options, allocator, &loop_sync));

  iree_loop_sync_scope_t* scope = NULL;
  IREE_CHECK_OK(
      iree_allocator_malloc(allocator, sizeof(*scope), (void**)&scope));
  iree_loop_sync_scope_initialize(
      loop_sync,
      +[](void* user_data, iree_status_t status) {
        iree_status_t* status_ptr = (iree_status_t*)user_data;
        if (iree_status_is_ok(*status_ptr)) {
          *status_ptr = status;
        } else {
          iree_status_ignore(status);
        }
      },
      out_status, scope);
  *out_loop = iree_loop_sync_scope(scope);
}

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop) {
  iree_loop_sync_scope_t* scope = (iree_loop_sync_scope_t*)loop.self;
  iree_loop_sync_t* loop_sync = scope->loop_sync;

  iree_loop_sync_scope_deinitialize(scope);
  iree_allocator_free(allocator, scope);

  iree_loop_sync_free(loop_sync);
}

// TODO(benvanik): test multiple scopes and scoped abort behavior.
