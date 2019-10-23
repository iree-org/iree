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

#ifndef IREE_RT_DISASSEMBLER_H_
#define IREE_RT_DISASSEMBLER_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/status.h"
#include "rt/function.h"
#include "rt/source_location.h"

namespace iree {
namespace rt {

// A single disassembled instruction.
struct Instruction {
  // Offset of the instruction within the function.
  // The meaning of this is backend-dependent.
  SourceOffset offset;

  // The first line of |long_text|.
  absl::string_view short_text;

  // Human-readable text of the instruction. May contain multiple lines.
  std::string long_text;
};

// Disassembles functions into instructions.
//
// Thread-safe.
class Disassembler {
 public:
  virtual ~Disassembler() = default;

  // Disassembles one or more instructions within the given function based on
  // source offsets.
  virtual StatusOr<std::vector<Instruction>> DisassembleInstructions(
      const Function& function, SourceOffset offset,
      int32_t instruction_count = INT32_MAX) const = 0;

 protected:
  Disassembler() = default;
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DISASSEMBLER_H_
