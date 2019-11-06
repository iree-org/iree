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

# Samples demonstrating use of the RT API.

load("//iree:build_defs.bzl", "PLATFORM_VULKAN_TEST_DEPS")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "simple_module_test_bytecode_module",
    srcs = ["simple_module_test.mlir"],
    cc_namespace = "iree::rt::samples",
)

cc_test(
    name = "bytecode_module_test",
    srcs = ["bytecode_module_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_absl//absl/strings",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/hal:buffer_view",
        "//iree/hal:driver_registry",
        "//iree/rt",
        "//iree/schemas",
        "//iree/vm:bytecode_module",
        "//iree/vm:sequencer_module",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "//iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)

cc_test(
    name = "bytecode_module_api_test",
    srcs = ["bytecode_module_api_test.cc"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        ":simple_module_test_bytecode_module_cc",
        "@com_google_absl//absl/strings",
        "//iree/base:api",
        "//iree/hal:api",
        "//iree/hal:driver_registry",
        "//iree/rt:api",
        "//iree/vm:api",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "//iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)
