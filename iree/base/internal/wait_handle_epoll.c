// Copyright 2020 Google LLC
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

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/wait_handle_impl.h"

#if IREE_WAIT_API == IREE_WAIT_API_EPOLL

#include "iree/base/internal/wait_handle_posix.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): iree_wait_set_s using an epoll fd.
// epoll lets us route the wait set operations right to kernel and not need our
// own duplicate data structure. epoll is great, just not available on mac/ios
// so we still need poll for that. linux/android/bsd all have epoll, though.
struct iree_wait_set_s {
  // NOTE: we could in theory use the epoll handle directly (iree_wait_set_s
  // then is just a pointer). Then allocate/free just go straight to the system.
  int reserved;
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  // TODO(benvanik): epoll_create()
}

void iree_wait_set_free(iree_wait_set_t* set) {
  // TODO(benvanik): close()
}

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  // TODO(benvanik): epoll_ctl(EPOLL_CTL_ADD)
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {
  // TODO(benvanik): epoll_ctl(EPOLL_CTL_DEL)
}

void iree_wait_set_clear(iree_wait_set_t* set) {
  // TODO(benvanik): close and reopen?
}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  // TODO(benvanik): epoll_wait
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  // TODO(benvanik): epoll_wait
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  // TODO(benvanik): just use poll?
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_EPOLL
