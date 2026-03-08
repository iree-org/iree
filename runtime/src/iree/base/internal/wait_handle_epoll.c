// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/wait_handle_impl.h"

#if IREE_WAIT_API == IREE_WAIT_API_EPOLL

#include "iree/base/internal/wait_handle_posix.h"

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  // TODO(benvanik): just use poll?
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_EPOLL
