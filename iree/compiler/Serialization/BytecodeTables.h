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

#ifndef IREE_COMPILER_SERIALIZATION_BYTECODE_TABLES_H_
#define IREE_COMPILER_SERIALIZATION_BYTECODE_TABLES_H_

#include "iree/schemas/bytecode/interpreter_bytecode_v0.h"
#include "iree/schemas/bytecode/sequencer_bytecode_v0.h"
#include "mlir/Support/LLVM.h"
#include "third_party/llvm/llvm/include/llvm/ADT/Optional.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringRef.h"

namespace mlir {
namespace iree_compiler {

struct OpcodeInfo {
  const char* mnemonic = nullptr;
  iree::OpcodeFlagBitfield flags = iree::OpcodeFlagBitfield::kDefault;
  union {
    const char operands_value[8] = {0};
    const iree::OperandEncoding operands[8];
  };
};

// Returns an opcode - if found - for the given interpreter op.
llvm::Optional<iree::InterpreterOpcode> GetInterpreterOpcodeByName(
    StringRef name);

// Returns the info for the given interpreter opcode.
const OpcodeInfo& GetInterpreterOpcodeInfo(iree::InterpreterOpcode opcode);

// Returns an opcode - if found - for the given sequencer op.
llvm::Optional<iree::SequencerOpcode> GetSequencerOpcodeByName(StringRef name);

// Returns the info for the given sequencer opcode.
const OpcodeInfo& GetSequencerOpcodeInfo(iree::SequencerOpcode opcode);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_BYTECODE_TABLES_H_
