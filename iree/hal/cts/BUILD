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

# Conformance Test Suite (CTS) for HAL implementations.

load("//iree:build_defs.oss.bzl", "PLATFORM_VULKAN_TEST_DEPS")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "cts_test_base",
    testonly = True,
    hdrs = ["cts_test_base.h"],
    data = [
        # For AddressSanitizer when using Vulkan + a local Nvidia GPU
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        "//iree/base:status",
        "//iree/hal:driver_registry",

        # HAL driver modules.
        "//iree/hal/dylib:dylib_driver_module",  # build-cleaner: keep
        "//iree/hal/llvmjit:llvmjit_driver_module",  # build-cleaner: keep
        "//iree/hal/vmla:vmla_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/base:status",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "command_buffer_test",
    srcs = ["command_buffer_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/base:status",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "command_queue_test",
    srcs = ["command_queue_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/base:status",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "device_creation_test",
    srcs = ["device_creation_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)
