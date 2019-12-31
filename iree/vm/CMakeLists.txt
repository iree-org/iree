# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_cc_library(
  NAME
    bytecode_disassembler
  SRCS
    "bytecode_disassembler.cc"
  HDRS
    "bytecode_disassembler.h"
  DEPS
    absl::core_headers
    absl::inlined_vector
    absl::strings
    absl::span
    iree::base::status
    iree::schemas
    iree::schemas::bytecode::bytecode_v0
    iree::rt
    iree::vm::bytecode_util
    iree::vm::opcode_info
    iree::vm::source_map_resolver
    iree::vm::type
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_module
  SRCS
    "bytecode_module.cc"
  HDRS
    "bytecode_module.h"
  DEPS
    absl::memory
    iree::base::flatbuffer_util
    iree::base::status
    iree::rt
    iree::schemas
    iree::vm::bytecode_disassembler
    iree::vm::source_map_resolver
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_reader
  SRCS
    "bytecode_reader.cc"
  HDRS
    "bytecode_reader.h"
  DEPS
    absl::core_headers
    absl::inlined_vector
    iree::base::shape
    iree::base::status
    iree::hal::buffer_view
    iree::hal::heap_buffer
    iree::rt
    iree::schemas::bytecode::bytecode_v0
    iree::vm::bytecode_module
    iree::vm::type
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_tables_interpreter
  SRCS
    "bytecode_tables_interpreter.cc"
  HDRS
    "bytecode_tables_interpreter.h"
  DEPS
    iree::schemas::bytecode::interpreter_bytecode_v0
    iree::vm::opcode_info
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_util
  SRCS
    "bytecode_util.cc"
  HDRS
    "bytecode_util.h"
  DEPS
    absl::strings
    iree::schemas::bytecode::bytecode_v0
  PUBLIC
)

iree_cc_library(
  NAME
    bytecode_validator
  SRCS
    "bytecode_validator.cc"
  HDRS
    "bytecode_validator.h"
  DEPS
    iree::base::status
    iree::rt
    iree::schemas
  PUBLIC
)

iree_cc_library(
  NAME
    opcode_info
  HDRS
    "opcode_info.h"
  DEPS
    absl::strings
    absl::optional
    absl::span
    iree::schemas::bytecode::bytecode_v0
  PUBLIC
)

iree_cc_library(
  NAME
    source_map_resolver
  SRCS
    "source_map_resolver.cc"
  HDRS
    "source_map_resolver.h"
  DEPS
    absl::strings
    absl::optional
    iree::base::flatbuffer_util
    iree::base::status
    iree::rt
    iree::schemas
  PUBLIC
)

iree_cc_library(
  NAME
    type
  SRCS
    "type.cc"
  HDRS
    "type.h"
  DEPS
    iree::base::status
    iree::schemas
    iree::schemas::bytecode::bytecode_v0
  PUBLIC
)
