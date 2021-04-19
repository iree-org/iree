# Copyright 2021 Google LLC
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

# Tests of end-to-end IREE support for support of specific subsets of ops in the
# XLA HLO dialect. This is for cases where some variation (e.g. tensor element
# types) of an op is not supported on all backends. When the test is supported
# on all backends it should be moved into the corresponding op test in the main
# xla_ops direcotry. Each test file should have a name matching the
# corresponding XLA HLO op with a suffix indicating the subset it tests. Only
# the functionality of that op should be tessted (though it may make use of
# other ops where necessary). Tests should be written using the IREE Check
# framework and should always pass on the reference VMLA backend. See
# https://google.github.io/iree/TestingGuide#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_vulkan-spirv_vulkan_f16",
    srcs = [
        "add_f16.mlir",
        "dot_f16.mlir",
    ],
    compiler_flags = [
        "-iree-vulkan-target-triple=valhall-g77-unknown-android10",
    ],
    driver = "vulkan",
    tags = [
        "notap",
        "vulkan_uses_vk_khr_shader_float16_int8",
    ],
    target_backend = "vulkan-spirv",
)
