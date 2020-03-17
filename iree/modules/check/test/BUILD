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

load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = glob(["*.mlir"]),
    data = [
        "//iree/modules/check:iree-check-module",
        "//iree/modules/check/dialect:check-translate",
        "//iree/tools:IreeFileCheck",
    ],
)

iree_bytecode_module(
    name = "success_module",
    src = "success.mlir",
    flags = ["-iree-mlir-to-vm-bytecode-module"],
    translate_tool = "//iree/modules/check/dialect:check-translate",
)

sh_test(
    name = "success_test",
    srcs = ["//iree/modules/check:check_module.sh"],
    args = [
        "--driver=vmla",
        "--input_file=$(location :success_module)",
    ],
    data = [
        ":success_module",
        "//iree/modules/check:iree-check-module",
    ],
)

iree_bytecode_module(
    name = "failure_module",
    src = "failure.mlir",
    flags = ["-iree-mlir-to-vm-bytecode-module"],
    translate_tool = "//iree/modules/check/dialect:check-translate",
)

sh_test(
    name = "failure_test",
    srcs = ["//iree/modules/check:check_module.sh"],
    args = [
        "--driver=vmla",
        "--input_file=$(location :failure_module)",
        "--expect_failure",
    ],
    data = [
        ":failure_module",
        "//iree/modules/check:iree-check-module",
    ],
)
