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

load("//iree:build_defs.oss.bzl", "FLATBUFFER_SUPPORTS_REFLECTIONS", "iree_build_test", "iree_flatbuffer_cc_library")
load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")
load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
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

iree_flatbuffer_c_library(
    name = "bytecode_module_def_c_fbs",
    srcs = ["bytecode_module_def.fbs"],
    flatcc_args = [
        "--reader",
        "--builder",
        "--verifier",
    ],
)

iree_flatbuffer_cc_library(
    name = "bytecode_module_def_cc_fbs",
    srcs = ["bytecode_module_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_flatbuffer_cc_library(
    name = "dylib_executable_def_cc_fbs",
    srcs = ["dylib_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_flatbuffer_cc_library(
    name = "llvmir_executable_def_cc_fbs",
    srcs = ["llvmir_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_flatbuffer_cc_library(
    name = "spirv_executable_def_cc_fbs",
    srcs = ["spirv_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_flatbuffer_cc_library(
    name = "vmla_executable_def_cc_fbs",
    srcs = ["vmla_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
)

iree_build_test(
    name = "schema_build_test",
    targets = [
        ":bytecode_module_def_cc_fbs",
        ":dylib_executable_def_cc_fbs",
        ":llvmir_executable_def_cc_fbs",
        ":spirv_executable_def_cc_fbs",
        ":vmla_executable_def_cc_fbs",
    ],
)

REFLECTION_SRCS = [] if not FLATBUFFER_SUPPORTS_REFLECTIONS else [
    "bytecode_module_def.bfbs",
    "dylib_executable_def.bfbs",
    "llvmir_executable_def.bfbs",
    "spirv_executable_def.bfbs",
    "vmla_executable_def.bfbs",
]

cc_embed_data(
    name = "reflection_data",
    srcs = REFLECTION_SRCS,
    cc_file_output = "reflection_data.cc",
    cpp_namespace = "iree::schemas",
    h_file_output = "reflection_data.h",
)
