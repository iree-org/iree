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

load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["bazel_toolchains.patch"])

bzl_library(
    name = "tblgen_bzl",
    srcs = ["tblgen.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "third_party_import/llvm-project/configure_bzl",
    srcs = ["third_party_import/llvm-project/configure.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "iree_tablegen_doc_bzl",
    srcs = ["iree_tablegen_doc.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "run_binary_test_bzl",
    srcs = ["run_binary_test.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "iree_check_test_bzl",
    srcs = ["iree_check_test.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "iree_flatcc_bzl",
    srcs = ["iree_flatcc.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "template_rule_bzl",
    srcs = ["template_rule.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "iree_py_test_suite_bzl",
    srcs = ["iree_py_test_suite.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
