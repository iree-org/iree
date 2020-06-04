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

# A VMLA (VM-based Linear Algebra) runtime HAL backend.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "op_kernels",
    hdrs = ["op_kernels.h"],
    textual_hdrs = [
        # TODO(benvanik): SIMD variants.
        "op_kernels_generic.h",
        "op_kernels_ruy.h",
    ],
    deps = [
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_ruy//ruy",
        "@com_google_ruy//ruy:context",
    ],
)

cc_test(
    name = "op_kernels_test",
    srcs = ["op_kernels_test.cc"],
    deps = [
        ":op_kernels",
        "//iree/base:memory",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "vmla_cache",
    srcs = ["vmla_cache.cc"],
    hdrs = ["vmla_cache.h"],
    deps = [
        ":vmla_executable",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_format",
        "//iree/vm:instance",
        "//iree/vm:module",
    ],
)

cc_library(
    name = "vmla_device",
    srcs = ["vmla_device.cc"],
    hdrs = ["vmla_device.h"],
    deps = [
        ":vmla_cache",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal/host:host_local_device",
        "//iree/vm:instance",
        "//iree/vm:module",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vmla_driver",
    srcs = ["vmla_driver.cc"],
    hdrs = ["vmla_driver.h"],
    deps = [
        ":vmla_device",
        ":vmla_module",
        "//iree/base:api_util",
        "//iree/base:tracing",
        "//iree/hal:device_info",
        "//iree/hal:driver",
        "//iree/hal/host:serial_scheduling_model",
        "//iree/vm:instance",
        "//iree/vm:module",
    ],
)

cc_library(
    name = "vmla_driver_module",
    srcs = ["vmla_driver_module.cc"],
    deps = [
        ":vmla_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "vmla_executable",
    srcs = ["vmla_executable.cc"],
    hdrs = ["vmla_executable.h"],
    deps = [
        ":vmla_module",
        "//iree/base:api_util",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/hal/host:host_buffer",
        "//iree/hal/host:host_executable",
        "//iree/schemas:vmla_executable_def_cc_fbs",
        "//iree/vm:bytecode_module",
        "//iree/vm:context",
        "//iree/vm:instance",
        "//iree/vm:module",
        "//iree/vm:variant_list",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vmla_module",
    srcs = ["vmla_module.cc"],
    hdrs = ["vmla_module.h"],
    deps = [
        ":op_kernels",
        "//iree/base:api",
        "//iree/base:memory",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/vm",
        "//iree/vm:module_abi_cc",
        "//iree/vm:types",
        "@com_google_absl//absl/types:span",
    ],
)
