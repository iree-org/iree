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

iree_add_all_subdirs()

flatbuffer_c_library(
  NAME
    bytecode_module_def_c_fbs
  SRCS
    "bytecode_module_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    cuda_executable_def_c_fbs
  SRCS
    "cuda_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    dylib_executable_def_c_fbs
  SRCS
    "dylib_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    metal_executable_def_c_fbs
  SRCS
    "metal_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    spirv_executable_def_c_fbs
  SRCS
    "spirv_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    vmla_executable_def_c_fbs
  SRCS
    "vmla_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)
