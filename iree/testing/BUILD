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

# Testing utilities for IREE.

load("//iree:build_defs.oss.bzl", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "benchmark_main",
    testonly = True,
    srcs = ["benchmark_main.cc"],
    deps = [
        "//iree/base:init",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "gtest",
    testonly = True,
    hdrs = ["gtest.h"],
    deps = platform_trampoline_deps("gtest", "testing"),
)

cc_library(
    name = "gtest_main",
    testonly = True,
    hdrs = ["gtest.h"],
    tags = ["keep_dep"],
    deps = platform_trampoline_deps("gtest_main", "testing"),
)
