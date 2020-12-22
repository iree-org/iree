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

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_test_suite")
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = glob(["*.mlir"]),
    data = [
        "//iree/modules/check:iree-check-module",
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-translate",
    ],
    tags = ["hostonly"],
)

iree_check_test_suite(
    name = "check",
    srcs = ["success.mlir"],
)

iree_check_test_suite(
    name = "check_failure",
    srcs = ["failure.mlir"],
    runner_args = ["--expect_failure"],
)
