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

# Bytecode VM used by the IREE interpreter.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_disassembler.cc",
        "bytecode_module.cc",
    ],
    hdrs = [
        "bytecode_disassembler.h",
        "bytecode_module.h",
    ],
    deps = [
        ":bytecode_util",
        ":opcode_info",
        ":source_map_resolver",
        ":type",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:buffer_view",
        "//iree/rt",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bytecode_reader",
    srcs = ["bytecode_reader.cc"],
    hdrs = ["bytecode_reader.h"],
    deps = [
        ":bytecode_module",
        ":type",
        "//iree/base:shape",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "//iree/hal:heap_buffer",
        "//iree/rt",
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "bytecode_tables_interpreter",
    srcs = ["bytecode_tables_interpreter.cc"],
    hdrs = ["bytecode_tables_interpreter.h"],
    deps = [
        ":opcode_info",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_util",
    srcs = ["bytecode_util.cc"],
    hdrs = ["bytecode_util.h"],
    deps = [
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bytecode_validator",
    srcs = ["bytecode_validator.cc"],
    hdrs = ["bytecode_validator.h"],
    deps = [
        ":bytecode_module",
        "//iree/base:status",
        "//iree/schemas",
    ],
)

cc_library(
    name = "opcode_info",
    hdrs = ["opcode_info.h"],
    deps = [
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "source_map_resolver",
    srcs = ["source_map_resolver.cc"],
    hdrs = ["source_map_resolver.h"],
    deps = [
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/rt",
        "//iree/schemas",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        "//iree/base:status",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
    ],
)
