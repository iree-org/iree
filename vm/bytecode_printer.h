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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_BYTECODE_PRINTER_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_BYTECODE_PRINTER_H_

#include <ostream>

#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/schemas/bytecode_def_generated.h"
#include "third_party/mlir_edge/iree/schemas/source_map_def_generated.h"
#include "third_party/mlir_edge/iree/vm/executable_table.h"
#include "third_party/mlir_edge/iree/vm/function_table.h"
#include "third_party/mlir_edge/iree/vm/opcode_info.h"
#include "third_party/mlir_edge/iree/vm/source_map.h"

namespace iree {
namespace vm {

// Prints bytecode in a text format to enable human-inspection.
// Optionally can interleave original source location information if a SourceMap
// is available.
class BytecodePrinter {
 public:
  static std::string ToString(OpcodeTable opcode_table,
                              const FunctionTable& function_table,
                              const ExecutableTable& executable_table,
                              const SourceMapResolver& source_map_resolver,
                              const BytecodeDef& bytecode_def);

  explicit BytecodePrinter(OpcodeTable opcode_table,
                           const FunctionTable& function_table,
                           const ExecutableTable& executable_table,
                           const SourceMapResolver& source_map_resolver)
      : opcode_table_(opcode_table),
        function_table_(function_table),
        executable_table_(executable_table),
        source_map_resolver_(source_map_resolver) {}

  StatusOr<std::string> Print(const BytecodeDef& bytecode_def) const;

  Status PrintToStream(const BytecodeDef& bytecode_def,
                       std::ostream* stream) const;
  Status PrintToStream(absl::Span<const uint8_t> data,
                       std::ostream* stream) const;

 private:
  OpcodeTable opcode_table_;
  const FunctionTable& function_table_;
  const ExecutableTable& executable_table_;
  const SourceMapResolver& source_map_resolver_;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_BYTECODE_PRINTER_H_
