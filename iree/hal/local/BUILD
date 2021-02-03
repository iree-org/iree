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

# Default implementations for HAL types that use the host resources.
# These are generally just wrappers around host heap memory and host threads.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(benvanik): replace iree/base/arena.h with this one. We still want the
# old-style arena for pure stack use; we may be able to do that with a change
# to block pool that allows for on-stack initialization (iree_stack_arena_t
# that has storage for one block inside itself and then dynamically allocates
# new ones if needed). That way we have only one arena implementation and can
# easily use the iree_allocator_t interface without worry.
cc_library(
    name = "arena",
    srcs = ["arena.c"],
    hdrs = ["arena.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:synchronization",
        "//iree/base/internal:atomic_slist",
    ],
)

# TODO(benvanik): move into base/? may be useful for other backends or for other
# parts of the system (like modules handling IO/RPC).
cc_library(
    name = "event_pool",
    srcs = ["event_pool.c"],
    hdrs = ["event_pool.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:synchronization",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:wait_handle",
    ],
)

cc_library(
    name = "executable_library",
    hdrs = ["executable_library.h"],
)

cc_library(
    name = "local",
    srcs = [
        "executable_loader.c",
        "local_descriptor_set.c",
        "local_descriptor_set_layout.c",
        "local_executable.c",
        "local_executable_cache.c",
        "local_executable_layout.c",
    ],
    hdrs = [
        "executable_loader.h",
        "local_descriptor_set.h",
        "local_descriptor_set_layout.h",
        "local_executable.h",
        "local_executable_cache.h",
        "local_executable_layout.h",
    ],
    deps = [
        ":executable_library",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/hal:api",
    ],
)

cc_library(
    name = "task_driver",
    srcs = [
        "task_command_buffer.c",
        "task_device.c",
        "task_driver.c",
        "task_event.c",
        "task_queue.c",
        "task_queue_state.c",
        "task_semaphore.c",
    ],
    hdrs = [
        "task_command_buffer.h",
        "task_device.h",
        "task_driver.h",
        "task_event.h",
        "task_queue.h",
        "task_queue_state.h",
        "task_semaphore.h",
    ],
    deps = [
        ":arena",
        ":event_pool",
        ":local",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:synchronization",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:wait_handle",
        "//iree/hal:api",
        "//iree/task",
    ],
)
