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

load("//iree:build_defs.bzl", "FLATBUFFER_SUPPORTS_REFLECTIONS", "iree_build_test", "iree_flatbuffer_cc_library")
load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

FLATC_ARGS = [
    # Preserve workspace-relative include paths in generated code.
    "--keep-prefix",
    # Use C++11 'enum class' for enums.
    "--scoped-enums",
    # Include reflection tables used for dumping debug representations.
    "--reflect-names",
    # Generate FooT types for unpack/pack support. Note that this should only
    # be used in tooling as the code size/runtime overhead is non-trivial.
    "--gen-object-api",
]

iree_flatbuffer_cc_library(
    name = "archive_def_cc_fbs",
    srcs = ["archive_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":module_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "buffer_data_def_cc_fbs",
    srcs = ["buffer_data_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_flatbuffer_cc_library(
    name = "bytecode_def_cc_fbs",
    srcs = ["bytecode_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

# TODO(benvanik): also expose as C using flatcc.
iree_flatbuffer_cc_library(
    name = "bytecode_module_def_cc_fbs",
    srcs = ["bytecode_module_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "debug_service_cc_fbs",
    srcs = ["debug_service.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":module_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "device_def_cc_fbs",
    srcs = ["device_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "device_group_def_cc_fbs",
    srcs = ["device_group_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "device_table_def_cc_fbs",
    srcs = ["device_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "executable_def_cc_fbs",
    srcs = ["executable_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "executable_table_def_cc_fbs",
    srcs = ["executable_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":executable_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "function_def_cc_fbs",
    srcs = ["function_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "function_table_def_cc_fbs",
    srcs = ["function_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "module_def_cc_fbs",
    srcs = ["module_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

iree_flatbuffer_cc_library(
    name = "source_map_def_cc_fbs",
    srcs = ["source_map_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "spirv_executable_def_cc_fbs",
    srcs = ["spirv_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_flatbuffer_cc_library(
    name = "type_def_cc_fbs",
    srcs = ["type_def.fbs"],
    flatc_args = FLATC_ARGS,
    includes = [
    ],
)

iree_build_test(
    name = "schema_build_test",
    targets = [
        ":archive_def_cc_fbs",
        ":bytecode_def_cc_fbs",
        ":debug_service_cc_fbs",
        ":device_def_cc_fbs",
        ":device_group_def_cc_fbs",
        ":device_table_def_cc_fbs",
        ":executable_def_cc_fbs",
        ":executable_table_def_cc_fbs",
        ":function_def_cc_fbs",
        ":function_table_def_cc_fbs",
        ":module_def_cc_fbs",
        ":source_map_def_cc_fbs",
        ":spirv_executable_def_cc_fbs",
        ":type_def_cc_fbs",
    ],
)

cc_library(
    name = "schemas",
    hdrs = [
        ":archive_def_generated.h",
        ":buffer_data_def_generated.h",
        ":bytecode_def_generated.h",
        ":debug_service_generated.h",
        ":device_def_generated.h",
        ":device_group_def_generated.h",
        ":device_table_def_generated.h",
        ":executable_def_generated.h",
        ":executable_table_def_generated.h",
        ":function_def_generated.h",
        ":function_table_def_generated.h",
        ":module_def_generated.h",
        ":source_map_def_generated.h",
        ":type_def_generated.h",
    ],
    deps = [
        ":archive_def_cc_fbs",
        ":buffer_data_def_cc_fbs",
        ":bytecode_def_cc_fbs",
        ":debug_service_cc_fbs",
        ":device_def_cc_fbs",
        ":device_group_def_cc_fbs",
        ":device_table_def_cc_fbs",
        ":executable_def_cc_fbs",
        ":executable_table_def_cc_fbs",
        ":function_def_cc_fbs",
        ":function_table_def_cc_fbs",
        ":module_def_cc_fbs",
        ":source_map_def_cc_fbs",
        ":spirv_executable_def_cc_fbs",
        ":type_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)

REFLECTION_SRCS = [] if not FLATBUFFER_SUPPORTS_REFLECTIONS else [
    "archive_def.bfbs",
    "buffer_data_def.bfbs",
    "bytecode_def.bfbs",
    "debug_service.bfbs",
    "executable_def.bfbs",
    "executable_table_def.bfbs",
    "function_def.bfbs",
    "function_table_def.bfbs",
    "module_def.bfbs",
    "source_map_def.bfbs",
    "spirv_executable_def.bfbs",
    "type_def.bfbs",
    "device_def.bfbs",
    "device_group_def.bfbs",
    "device_table_def.bfbs",
]

cc_embed_data(
    name = "reflection_data",
    srcs = REFLECTION_SRCS,
    cc_file_output = "reflection_data.cc",
    cpp_namespace = "iree::schemas",
    h_file_output = "reflection_data.h",
)
