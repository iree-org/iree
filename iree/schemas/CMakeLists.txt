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

add_subdirectory(bytecode)

flatbuffer_cc_library(
  NAME
    buffer_data_def_cc_fbs
  SRCS
    "buffer_data_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    bytecode_module_def_cc_fbs
  SRCS
    "bytecode_module_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    interpreter_module_def_cc_fbs
  SRCS
    "interpreter_module_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    spirv_executable_def_cc_fbs
  SRCS
    "spirv_executable_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    vmla_executable_def_cc_fbs
  SRCS
    "vmla_executable_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    llvmir_executable_def_cc_fbs
  SRCS
    "llvmir_executable_def.fbs"
  FLATC_ARGS
    "--keep-prefix"
    "--scoped-enums"
    "--reflect-names"
    "--gen-object-api"
  PUBLIC
)

iree_cc_embed_data(
  NAME
    reflection_data
  CC_FILE_OUTPUT
    "reflection_data.cc"
  H_FILE_OUTPUT
    "reflection_data.h"
  CPP_NAMESPACE
    "iree::schemas"
  PUBLIC
)
