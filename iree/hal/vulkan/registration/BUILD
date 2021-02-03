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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(${IREE_HAL_DRIVER_VULKAN})
""",
    inline = True,
)

cc_library(
    name = "registration",
    srcs = ["driver_module.cc"],
    hdrs = ["driver_module.h"],
    defines = [
        "IREE_HAL_HAVE_VULKAN_DRIVER_MODULE=1",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/base/internal:flags",
        "//iree/hal:api",
        "//iree/hal/vulkan",
        "@com_google_absl//absl/flags:flag",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
