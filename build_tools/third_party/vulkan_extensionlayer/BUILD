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
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "vulkan_loader_setup",
    hdrs = ["vulkan_loader_setup.h"],
    srcs = ["vulkan_loader_setup.cc"],
    deps = [
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@iree_core//iree/base:init",
        "@iree_core//iree/base:initializer",
        "@iree_core//iree/base:logging",
    ],
    data = [
        "@vulkan_extensionlayer//:libVkLayer_khronos_timeline_semaphore.so",
        "@vulkan_extensionlayer//:VkLayer_khronos_timeline_semaphore_files",
    ],
)
