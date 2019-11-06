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

# Tooling for benchmarking IREE.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "benchmark_module",
    srcs = ["benchmark_module.cc"],
    hdrs = ["benchmark_module.h"],
    deps = [
        "//iree/base:shaped_buffer",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "//iree/hal:driver_registry",
        "//iree/rt",
        "//iree/rt/debug:debug_server_flags",
        "//iree/vm:bytecode_module",
        "//iree/vm:sequencer_module",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "benchmark_iree_module",
    testonly = 1,
    srcs = ["benchmark_module_main.cc"],
    deps = [
        ":benchmark_module",
        "//iree/base:file_io",
        "//iree/base:init",
        "//iree/base:shaped_buffer_string_util",
        "//iree/base:status",
        "//iree/hal/interpreter:interpreter_driver_module",
        "//iree/vm:bytecode_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
    ],
)
