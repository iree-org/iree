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

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "dynamic_library_test_library.so",
    testonly = True,
    srcs = ["dynamic_library_test_library.cc"],
    linkshared = True,
)

cc_embed_data(
    name = "dynamic_library_test_library",
    testonly = True,
    srcs = [":dynamic_library_test_library.so"],
    cc_file_output = "dynamic_library_test_library_embed.cc",
    cpp_namespace = "iree",
    flatten = True,
    h_file_output = "dynamic_library_test_library_embed.h",
)

cc_test(
    name = "dynamic_library_test",
    srcs = ["dynamic_library_test.cc"],
    deps = [
        ":dynamic_library_test_library",
        "//iree/base:core_headers",
        "//iree/base:dynamic_library",
        "//iree/base:status",
        "//iree/base/internal:file_io",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
