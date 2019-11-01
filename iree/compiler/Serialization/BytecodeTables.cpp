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

#include "iree/compiler/Serialization/BytecodeTables.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Info tables mapping 1:1 with bytecode ops.
//
// Note that we ensure the table is 256 elements long exactly to make sure
// that unused opcodes are handled gracefully.
#define DECLARE_INFO(ordinal, enum_value, name, flags, operand_encodings, ...) \
  {                                                                            \
      name,                                                                    \
      flags,                                                                   \
      {operand_encodings},                                                     \
  },

static const OpcodeInfo kInterpreterInfoTable[256] = {
    IREE_INTERPRETER_OPCODE_LIST(DECLARE_INFO, DECLARE_INFO)};

static const OpcodeInfo kSequencerInfoTable[256] = {
    IREE_SEQUENCER_OPCODE_LIST(DECLARE_INFO, DECLARE_INFO)};

#undef DECLARE_INFO

}  // namespace

llvm::Optional<iree::InterpreterOpcode> GetInterpreterOpcodeByName(
    StringRef name) {
  for (int i = 0; i < llvm::array_lengthof(kInterpreterInfoTable); ++i) {
    if (name == kInterpreterInfoTable[i].mnemonic) {
      return static_cast<iree::InterpreterOpcode>(i);
    }
  }
  return llvm::None;
}

const OpcodeInfo& GetInterpreterOpcodeInfo(iree::InterpreterOpcode opcode) {
  return kInterpreterInfoTable[static_cast<uint8_t>(opcode)];
}

llvm::Optional<iree::SequencerOpcode> GetSequencerOpcodeByName(StringRef name) {
  for (int i = 0; i < llvm::array_lengthof(kSequencerInfoTable); ++i) {
    if (name == kSequencerInfoTable[i].mnemonic) {
      return static_cast<iree::SequencerOpcode>(i);
    }
  }
  return llvm::None;
}

const OpcodeInfo& GetSequencerOpcodeInfo(iree::SequencerOpcode opcode) {
  return kSequencerInfoTable[static_cast<uint8_t>(opcode)];
}

}  // namespace iree_compiler
}  // namespace mlir
