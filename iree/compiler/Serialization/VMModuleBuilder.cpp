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

#include "iree/compiler/Serialization/VMModuleBuilder.h"

namespace mlir {
namespace iree_compiler {

VMModuleBuilder::VMModuleBuilder(::flatbuffers::FlatBufferBuilder *fbb)
    : fbb_(fbb), functionTable_(fbb) {}

::flatbuffers::Offset<iree::ModuleDef> VMModuleBuilder::Finish() {
  auto nameOffset = fbb_->CreateString("module");
  auto functionTableOffset = functionTable_.Finish();
  if (functionTableOffset.IsNull()) return {};

  iree::ModuleDefBuilder mdb(*fbb_);
  mdb.add_name(nameOffset);
  mdb.add_function_table(functionTableOffset);
  return mdb.Finish();
}

std::vector<uint8_t> VMModuleBuilder::Serialize(
    ::flatbuffers::Offset<iree::ModuleDef> module_def) {
  FinishModuleDefBuffer(*fbb_, module_def);
  std::vector<uint8_t> bytes;
  bytes.resize(fbb_->GetSize());
  std::memcpy(bytes.data(), fbb_->GetBufferPointer(), bytes.size());
  return bytes;
}

}  // namespace iree_compiler
}  // namespace mlir
