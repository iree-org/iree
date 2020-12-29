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

objc_library(
    name = "metal",
    srcs = [
        "metal_pipeline_argument_buffer.cc",
    ],
    hdrs = [
        "dispatch_time_util.h",
        "metal_buffer.h",
        "metal_capture_manager.h",
        "metal_command_buffer.h",
        "metal_command_queue.h",
        "metal_device.h",
        "metal_direct_allocator.h",
        "metal_driver.h",
        "metal_kernel_library.h",
        "metal_pipeline_argument_buffer.h",
        "metal_pipeline_cache.h",
        "metal_shared_event.h",
    ],
    copts = ["-std=c++14"],
    non_arc_srcs = [
        "metal_buffer.mm",
        "metal_capture_manager.mm",
        "metal_command_buffer.mm",
        "metal_command_queue.mm",
        "metal_device.mm",
        "metal_direct_allocator.mm",
        "metal_driver.mm",
        "metal_kernel_library.mm",
        "metal_pipeline_cache.mm",
        "metal_shared_event.mm",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        "//iree/base:arena",
        "//iree/base:file_io",
        "//iree/base:flatcc",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal",
        "//iree/hal:command_buffer_validation",
        "//iree/schemas:metal_executable_def_c_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)
