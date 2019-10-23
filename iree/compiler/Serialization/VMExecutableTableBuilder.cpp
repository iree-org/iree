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

#include "iree/compiler/Serialization/VMExecutableTableBuilder.h"

namespace mlir {
namespace iree_compiler {

VMExecutableTableBuilder::VMExecutableTableBuilder(
    ::flatbuffers::FlatBufferBuilder *fbb)
    : fbb_(fbb) {}

LogicalResult VMExecutableTableBuilder::AddMultiArchExecutable(
    ::flatbuffers::Offset<iree::MultiArchExecutableDef>
        multiArchExecutableDef) {
  multiArchExecutableDefs_.push_back(multiArchExecutableDef);
  return success();
}

::flatbuffers::Offset<iree::ExecutableTableDef>
VMExecutableTableBuilder::Finish() {
  auto multiArchExecutablesOffset =
      fbb_->CreateVector(multiArchExecutableDefs_);
  iree::ExecutableTableDefBuilder etdb(*fbb_);
  etdb.add_multi_arch_executables(multiArchExecutablesOffset);
  return etdb.Finish();
}

}  // namespace iree_compiler
}  // namespace mlir
