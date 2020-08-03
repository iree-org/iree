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
load(":build_defs.oss.bzl", "iree_py_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_py_library(
    name = "pathsetup",
    imports = ["."],
)

bzl_library(
    name = "build_defs.google_bzl",
    srcs = ["build_defs.google.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "build_defs.oss_bzl",
    srcs = ["build_defs.oss.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "build_tools/python/build_defs_bzl",
    srcs = ["build_tools/python/build_defs.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "build_tools/python/configure_bzl",
    srcs = ["build_tools/python/configure.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
