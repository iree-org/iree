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

iree_cc_library(
  NAME
    bytecode_v0
  HDRS
    "bytecode_v0.h"
  DEPS
    absl::core_headers
    iree::base::bitfield
  PUBLIC
)

iree_cc_library(
  NAME
    interpreter_bytecode_v0
  HDRS
    "interpreter_bytecode_v0.h"
  DEPS
    iree::schemas::bytecode::bytecode_v0
  PUBLIC
)

iree_cc_library(
  NAME
    sequencer_bytecode_v0
  HDRS
    "sequencer_bytecode_v0.h"
  DEPS
    iree::schemas::bytecode::bytecode_v0
  PUBLIC
)
