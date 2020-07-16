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

# Implementations for iree/testing/

load("//iree:build_defs.oss.bzl", "iree_build_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gtest_internal",
    testonly = True,
    hdrs = ["gtest_internal.h"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gtest_main_internal",
    testonly = True,
    srcs = ["gtest_main_internal.cc"],
    hdrs = ["gtest_internal.h"],
    deps = [
        "//iree/base:init",
        "@com_google_googletest//:gtest",
    ],
)

iree_build_test(
    name = "internal_build_test",
    targets = [
        ":gtest_internal",
        ":gtest_main_internal",
    ],
)
