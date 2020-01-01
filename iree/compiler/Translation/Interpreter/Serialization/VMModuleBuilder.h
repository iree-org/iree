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

#ifndef IREE_COMPILER_SERIALIZATION_VM_MODULE_BUILDER_H_
#define IREE_COMPILER_SERIALIZATION_VM_MODULE_BUILDER_H_

#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Translation/Interpreter/Serialization/VMFunctionTableBuilder.h"
#include "iree/schemas/interpreter_module_def_generated.h"

namespace mlir {
namespace iree_compiler {

class VMModuleBuilder {
 public:
  explicit VMModuleBuilder(::flatbuffers::FlatBufferBuilder *fbb);

  ::flatbuffers::FlatBufferBuilder *fbb() const { return fbb_; }
  VMFunctionTableBuilder *function_table() { return &functionTable_; }

  ::flatbuffers::Offset<iree::ModuleDef> Finish();

  std::vector<uint8_t> Serialize(
      ::flatbuffers::Offset<iree::ModuleDef> module_def);

 private:
  ::flatbuffers::FlatBufferBuilder *fbb_;

  VMFunctionTableBuilder functionTable_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_MODULE_BUILDER_H_
