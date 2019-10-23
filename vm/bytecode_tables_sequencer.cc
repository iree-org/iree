// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vm/bytecode_tables_sequencer.h"

#include "schemas/bytecode/sequencer_bytecode_v0.h"

namespace iree {
namespace vm {

namespace {

// Info table mapping 1:1 with bytecode ops.
//
// Note that we ensure the table is 256 elements long exactly to make sure
// that unused opcodes are handled gracefully.
static const OpcodeInfo kInfoTable[256] = {
#define DECLARE_INFO(ordinal, enum_value, name, flags, operand_encodings, ...) \
  OpcodeInfo{                                                                  \
      name,                                                                    \
      flags,                                                                   \
      {operand_encodings},                                                     \
  },
    IREE_SEQUENCER_OPCODE_LIST(DECLARE_INFO, DECLARE_INFO)
#undef DECLARE_INFO
};

}  // namespace

OpcodeTable sequencer_opcode_table() { return kInfoTable; }

}  // namespace vm
}  // namespace iree
