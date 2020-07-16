# Copyright 2019 Google LLC
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

# Test utilities for HAL-specific code.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mock_allocator",
    testonly = True,
    hdrs = ["mock_allocator.h"],
    deps = [
        "//iree/hal:allocator",
        "//iree/testing:gtest",
    ],
)

cc_library(
    name = "mock_command_buffer",
    testonly = True,
    hdrs = ["mock_command_buffer.h"],
    deps = [
        "//iree/hal:command_buffer",
        "//iree/testing:gtest",
    ],
)

cc_library(
    name = "mock_command_queue",
    testonly = True,
    hdrs = ["mock_command_queue.h"],
    deps = [
        "//iree/hal:command_queue",
        "//iree/testing:gtest",
    ],
)
