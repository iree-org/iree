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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "vulkan_gui_util",
    srcs = ["vulkan_gui_util.cc"],
    hdrs = ["vulkan_gui_util.h"],
    tags = [
        "manual",
        "nokokoro",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/hal/vulkan:api",
        "@dear_imgui",
        "@dear_imgui//:imgui_sdl_vulkan",
        "@iree_vulkan_headers//:vulkan_headers_no_prototypes",
        "@sdl2//:SDL2",
        "@vulkan_sdk//:sdk",
    ],
)
