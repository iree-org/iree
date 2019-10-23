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

#include "compiler/Serialization/VMDeviceTableBuilder.h"
#include "compiler/Serialization/VMExecutableTableBuilder.h"
#include "compiler/Serialization/VMFunctionTableBuilder.h"
#include "compiler/Serialization/VMSourceMapBuilder.h"
#include "flatbuffers/flatbuffers.h"
#include "schemas/module_def_generated.h"

namespace mlir {
namespace iree_compiler {

class VMModuleBuilder {
 public:
  explicit VMModuleBuilder(::flatbuffers::FlatBufferBuilder *fbb);

  ::flatbuffers::FlatBufferBuilder *fbb() const { return fbb_; }
  VMDeviceTableBuilder *device_table() { return &deviceTable_; }
  VMFunctionTableBuilder *function_table() { return &functionTable_; }
  VMExecutableTableBuilder *executable_table() { return &executableTable_; }
  VMSourceMapBuilder *source_map() { return &sourceMap_; }

  ::flatbuffers::Offset<iree::ModuleDef> Finish();

  std::vector<uint8_t> Serialize(
      ::flatbuffers::Offset<iree::ModuleDef> module_def);

 private:
  ::flatbuffers::FlatBufferBuilder *fbb_;

  VMDeviceTableBuilder deviceTable_;
  VMFunctionTableBuilder functionTable_;
  VMExecutableTableBuilder executableTable_;
  VMSourceMapBuilder sourceMap_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_MODULE_BUILDER_H_
