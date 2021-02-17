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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_CUDA})
  return()
endif()
""",
)

cc_library(
    name = "dynamic_symbols",
    srcs = [
        "cuda_headers.h",
        "dynamic_symbols.cc",
        "dynamic_symbols_tables.h",
    ],
    hdrs = [
        "dynamic_symbols.h",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:dynamic_library",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/types:span",
        "@cuda_headers",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    tags = ["driver=cuda"],
    deps = [
        ":dynamic_symbols",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
