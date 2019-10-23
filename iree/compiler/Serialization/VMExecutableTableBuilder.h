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

#ifndef IREE_COMPILER_SERIALIZATION_VM_EXECUTABLE_TABLE_BUILDER_H_
#define IREE_COMPILER_SERIALIZATION_VM_EXECUTABLE_TABLE_BUILDER_H_

#include "flatbuffers/flatbuffers.h"
#include "iree/schemas/executable_table_def_generated.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

class VMExecutableTableBuilder {
 public:
  explicit VMExecutableTableBuilder(::flatbuffers::FlatBufferBuilder *fbb);

  LogicalResult AddMultiArchExecutable(
      ::flatbuffers::Offset<iree::MultiArchExecutableDef>
          multiArchExecutableDef);

  ::flatbuffers::Offset<iree::ExecutableTableDef> Finish();

 private:
  ::flatbuffers::FlatBufferBuilder *fbb_;
  std::vector<::flatbuffers::Offset<iree::MultiArchExecutableDef>>
      multiArchExecutableDefs_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_EXECUTABLE_TABLE_BUILDER_H_
