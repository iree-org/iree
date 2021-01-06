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
    ],
)
