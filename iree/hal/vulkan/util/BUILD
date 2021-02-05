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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "arena",
    srcs = ["arena.cc"],
    hdrs = ["arena.h"],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "arena_test",
    srcs = ["arena_test.cc"],
    deps = [
        ":arena",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "intrusive_list",
    hdrs = [
        "intrusive_list.h",
        "intrusive_list_unique_ptr.inc",
    ],
    deps = [
        "//iree/base:logging",
    ],
)

cc_test(
    name = "intrusive_list_test",
    srcs = [
        "intrusive_list_test.cc",
        "intrusive_list_unique_ptr_test.cc",
    ],
    deps = [
        ":intrusive_list",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "ref_ptr",
    hdrs = ["ref_ptr.h"],
    deps = [
        "//iree/base:logging",
        "//iree/base/internal",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "ref_ptr_test",
    size = "small",
    srcs = ["ref_ptr_test.cc"],
    deps = [
        ":ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
