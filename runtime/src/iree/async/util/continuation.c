// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/continuation.h"

iree_host_size_t iree_async_continuation_dispatch(
    iree_async_continuation_submit_fn_t submit_fn, void* submit_context,
    iree_async_continuation_list_t* continuations, iree_status_t status) {
  iree_host_size_t continuation_count = continuations->count;
  if (continuation_count == 0) {
    // No continuations - caller still owns the status for their callback.
    // Do not consume it here.
    return 0;
  }

  iree_host_size_t continuation_start = continuations->start;
  iree_async_operation_t** continuation_operations = continuations->operations;

  // Clear count before potentially recursive submit. This prevents
  // double-dispatch if the submit triggers synchronous completions that
  // re-enter this function.
  continuations->count = 0;

  if (iree_status_is_ok(status)) {
    // Triggering operation succeeded - submit continuations.
    iree_async_operation_list_t continuation_list = {
        .values = continuation_operations + continuation_start,
        .count = continuation_count,
    };
    iree_status_t submit_status = submit_fn(submit_context, continuation_list);
    if (!iree_status_is_ok(submit_status)) {
      // Submit failed - invoke callbacks directly with the error.
      iree_host_size_t invoked = 0;
      for (iree_host_size_t i = 0; i < continuation_count; ++i) {
        iree_async_operation_t* operation =
            continuation_operations[continuation_start + i];
        if (operation->completion_fn) {
          operation->completion_fn(operation->user_data, operation,
                                   iree_status_clone(submit_status),
                                   IREE_ASYNC_COMPLETION_FLAG_NONE);
          ++invoked;
        }
      }
      iree_status_ignore(submit_status);
      return invoked;
    }
    // Operations submitted - completions will be counted via CQEs.
    return 0;
  } else {
    // Triggering operation failed or was cancelled - invoke continuation
    // callbacks directly with CANCELLED.
    //
    // We don't submit-then-cancel because some operations (like NOP) complete
    // immediately in the kernel before cancel can take effect. Invoking
    // callbacks directly ensures continuations consistently receive CANCELLED.
    //
    // Note: We do NOT consume the incoming status here - the caller still
    // owns it and will use it for the triggering operation's callback.
    iree_host_size_t invoked = 0;
    for (iree_host_size_t i = 0; i < continuation_count; ++i) {
      iree_async_operation_t* operation =
          continuation_operations[continuation_start + i];
      if (operation->completion_fn) {
        iree_status_t cancelled_status =
            iree_make_status(IREE_STATUS_CANCELLED, "chain cancelled");
        operation->completion_fn(operation->user_data, operation,
                                 cancelled_status,
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
        ++invoked;
      }
    }
    return invoked;
  }
}
