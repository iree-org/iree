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

load("//iree:build_defs.oss.bzl", "iree_build_test")
load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

FLATCC_ARGS = [
    "--reader",
    "--builder",
    "--verifier",
    "--json",
]

iree_flatbuffer_c_library(
    name = "bytecode_module_def_c_fbs",
    srcs = ["bytecode_module_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_flatbuffer_c_library(
    name = "cuda_executable_def_c_fbs",
    srcs = ["cuda_executable_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_flatbuffer_c_library(
    name = "rocm_executable_def_c_fbs",
    srcs = ["rocm_executable_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_flatbuffer_c_library(
    name = "dylib_executable_def_c_fbs",
    srcs = ["dylib_executable_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_flatbuffer_c_library(
    name = "metal_executable_def_c_fbs",
    srcs = ["metal_executable_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_flatbuffer_c_library(
    name = "spirv_executable_def_c_fbs",
    srcs = ["spirv_executable_def.fbs"],
    flatcc_args = FLATCC_ARGS,
)

iree_build_test(
    name = "schema_build_test",
    targets = [
        ":bytecode_module_def_c_fbs",
        ":dylib_executable_def_c_fbs",
        ":metal_executable_def_c_fbs",
        ":spirv_executable_def_c_fbs",
    ],
)
