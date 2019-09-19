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

#include "iree/vm/module.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

// static
Status Module::ValidateStructure(const ModuleDef& module_def) {
  // Must have a function table.
  if (module_def.function_table()) {
    RETURN_IF_ERROR(
        FunctionTable::ValidateStructure(*module_def.function_table()));
  } else {
    return InvalidArgumentErrorBuilder(ABSL_LOC)
           << "ModuleDef is missing a function table";
  }

  // May optionally have an executable table.
  if (module_def.executable_table()) {
    RETURN_IF_ERROR(
        ExecutableTable::ValidateStructure(*module_def.executable_table()));
  }

  return OkStatus();
}

// static
StatusOr<std::unique_ptr<Module>> Module::FromDef(const ModuleDef& module_def) {
  ASSIGN_OR_RETURN(auto module_file, ModuleFile::Create(&module_def, []() {}));
  return FromFile(std::move(module_file));
}

// static
StatusOr<std::unique_ptr<Module>> Module::FromFile(
    std::unique_ptr<ModuleFile> module_file) {
  if (module_file->root() == nullptr) {
    return InvalidArgumentErrorBuilder(ABSL_LOC) << "No root ModuleDef present";
  }
  const auto& module_def = *module_file->root();

  // Validates the structure of the module (but not bytecode).
  // This ensures we don't have flatbuffer vectors will null entries, etc.
  RETURN_IF_ERROR(Module::ValidateStructure(module_def));

  auto module = absl::WrapUnique(new Module(std::move(module_file)));

  // TODO(benvanik): validate internals here? or make explicit?

  return {std::move(module)};
}

Module::Module(std::unique_ptr<ModuleFile> module_file)
    : module_file_(std::move(module_file)),
      module_def_(*module_file_->root()),
      function_table_(*this, *module_def_.function_table()),
      executable_table_(*module_def_.executable_table()) {}

Module::~Module() = default;

}  // namespace vm
}  // namespace iree
