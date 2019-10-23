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

#ifndef IREE_VM_BYTECODE_DISASSEMBLER_H_
#define IREE_VM_BYTECODE_DISASSEMBLER_H_

#include <ostream>

#include "iree/base/status.h"
#include "iree/rt/disassembler.h"
#include "iree/schemas/bytecode_def_generated.h"
#include "iree/schemas/source_map_def_generated.h"
#include "iree/vm/opcode_info.h"

namespace iree {
namespace vm {

// Disassembles bytecode with a specific op set to text.
class BytecodeDisassembler final : public rt::Disassembler {
 public:
  explicit BytecodeDisassembler(OpcodeTable opcode_table)
      : opcode_table_(opcode_table) {}

  StatusOr<std::vector<rt::Instruction>> DisassembleInstructions(
      const rt::Function& function, rt::SourceOffset offset,
      int32_t instruction_count) const override;

 private:
  OpcodeTable opcode_table_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_BYTECODE_DISASSEMBLER_H_
