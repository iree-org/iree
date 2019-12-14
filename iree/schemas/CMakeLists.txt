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

iree_cc_library(
  NAME
    schemas
  HDRS
    "archive_def_generated.h"
    "buffer_data_def_generated.h"
    "bytecode_def_generated.h"
    "debug_service_generated.h"
    "device_def_generated.h"
    "device_group_def_generated.h"
    "device_table_def_generated.h"
    "executable_def_generated.h"
    "executable_table_def_generated.h"
    "function_def_generated.h"
    "function_table_def_generated.h"
    "module_def_generated.h"
    "source_map_def_generated.h"
    "type_def_generated.h"
  DEPS
    iree::schemas::archive_def_cc_fbs
    iree::schemas::buffer_data_def_cc_fbs
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::debug_service_cc_fbs
    iree::schemas::device_def_cc_fbs
    iree::schemas::device_group_def_cc_fbs
    iree::schemas::device_table_def_cc_fbs
    iree::schemas::executable_def_cc_fbs
    iree::schemas::executable_table_def_cc_fbs
    iree::schemas::function_def_cc_fbs
    iree::schemas::function_table_def_cc_fbs
    iree::schemas::module_def_cc_fbs
    iree::schemas::source_map_def_cc_fbs
    iree::schemas::spirv_executable_def_cc_fbs
    iree::schemas::type_def_cc_fbs
    flatbuffers
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    archive_def_cc_fbs
  SRCS
    "archive_def.fbs"
  DEPS
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::device_def_cc_fbs
    iree::schemas::device_group_def_cc_fbs
    iree::schemas::device_table_def_cc_fbs
    iree::schemas::executable_def_cc_fbs
    iree::schemas::executable_table_def_cc_fbs
    iree::schemas::function_def_cc_fbs
    iree::schemas::function_table_def_cc_fbs
    iree::schemas::module_def_cc_fbs
    iree::schemas::source_map_def_cc_fbs
    iree::schemas::type_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    buffer_data_def_cc_fbs
  SRCS
    "buffer_data_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    bytecode_def_cc_fbs
  SRCS
    "bytecode_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    bytecode_module_def_cc_fbs
  SRCS
    "bytecode_module_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    debug_service_cc_fbs
  SRCS
    "debug_service.fbs"
  DEPS
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::device_def_cc_fbs
    iree::schemas::device_group_def_cc_fbs
    iree::schemas::device_table_def_cc_fbs
    iree::schemas::executable_def_cc_fbs
    iree::schemas::executable_table_def_cc_fbs
    iree::schemas::function_def_cc_fbs
    iree::schemas::function_table_def_cc_fbs
    iree::schemas::module_def_cc_fbs
    iree::schemas::source_map_def_cc_fbs
    iree::schemas::type_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    device_def_cc_fbs
  SRCS
    "device_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    device_group_def_cc_fbs
  SRCS
    "device_group_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    device_table_def_cc_fbs
  SRCS
    "device_table_def.fbs"
  DEPS
    iree::schemas::device_def_cc_fbs
    iree::schemas::device_group_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    executable_def_cc_fbs
  SRCS
    "executable_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    executable_table_def_cc_fbs
  SRCS
    "executable_table_def.fbs"
  DEPS
    iree::schemas::executable_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    function_def_cc_fbs
  SRCS
    "function_def.fbs"
  DEPS
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::type_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    function_table_def_cc_fbs
  SRCS
    "function_table_def.fbs"
  DEPS
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::function_def_cc_fbs
    iree::schemas::type_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    module_def_cc_fbs
  SRCS
    "module_def.fbs"
  DEPS
    iree::schemas::bytecode_def_cc_fbs
    iree::schemas::device_def_cc_fbs
    iree::schemas::device_group_def_cc_fbs
    iree::schemas::device_table_def_cc_fbs
    iree::schemas::executable_def_cc_fbs
    iree::schemas::executable_table_def_cc_fbs
    iree::schemas::function_def_cc_fbs
    iree::schemas::function_table_def_cc_fbs
    iree::schemas::source_map_def_cc_fbs
    iree::schemas::type_def_cc_fbs
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    source_map_def_cc_fbs
  SRCS
    "source_map_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    spirv_executable_def_cc_fbs
  SRCS
    "spirv_executable_def.fbs"
  PUBLIC
)

flatbuffer_cc_library(
  NAME
    type_def_cc_fbs
  SRCS
    "type_def.fbs"
  PUBLIC
)
