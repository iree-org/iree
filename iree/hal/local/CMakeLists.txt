# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

iree_cc_library(
  NAME
    arena
  HDRS
    "arena.h"
  SRCS
    "arena.c"
  DEPS
    iree::base::api
    iree::base::core_headers
    iree::base::internal::atomic_slist
    iree::base::synchronization
  PUBLIC
)

iree_cc_library(
  NAME
    event_pool
  HDRS
    "event_pool.h"
  SRCS
    "event_pool.c"
  DEPS
    iree::base::api
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::wait_handle
    iree::base::synchronization
    iree::base::tracing
  PUBLIC
)

iree_cc_library(
  NAME
    executable_library
  HDRS
    "executable_library.h"
  PUBLIC
)

iree_cc_library(
  NAME
    local
  HDRS
    "executable_loader.h"
    "local_descriptor_set.h"
    "local_descriptor_set_layout.h"
    "local_executable.h"
    "local_executable_cache.h"
    "local_executable_layout.h"
  SRCS
    "executable_loader.c"
    "local_descriptor_set.c"
    "local_descriptor_set_layout.c"
    "local_executable.c"
    "local_executable_cache.c"
    "local_executable_layout.c"
  DEPS
    ::executable_library
    iree::base::api
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal::api
  PUBLIC
)

iree_cc_library(
  NAME
    task_driver
  HDRS
    "task_command_buffer.h"
    "task_device.h"
    "task_driver.h"
    "task_event.h"
    "task_queue.h"
    "task_queue_state.h"
    "task_semaphore.h"
  SRCS
    "task_command_buffer.c"
    "task_device.c"
    "task_driver.c"
    "task_event.c"
    "task_queue.c"
    "task_queue_state.c"
    "task_semaphore.c"
  DEPS
    ::arena
    ::event_pool
    ::local
    iree::base::api
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::wait_handle
    iree::base::synchronization
    iree::base::tracing
    iree::hal::api
    iree::task
  PUBLIC
)
