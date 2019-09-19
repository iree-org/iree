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

#ifndef IREE_VM_MODULE_H_
#define IREE_VM_MODULE_H_

#include <memory>

#include "iree/base/flatbuffer_util.h"
#include "iree/schemas/module_def_generated.h"
#include "iree/vm/executable_table.h"
#include "iree/vm/function_table.h"

namespace iree {
namespace vm {

using ModuleFile = FlatBufferFile<ModuleDef>;

// A loaded bytecode module.
class Module {
 public:
  static Status ValidateStructure(const ModuleDef& module_def);

  static StatusOr<std::unique_ptr<Module>> FromDef(const ModuleDef& module_def);
  static StatusOr<std::unique_ptr<Module>> FromFile(
      std::unique_ptr<ModuleFile> module_file);

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  ~Module();

  absl::string_view name() const { return WrapString(module_def_.name()); }

  const ModuleDef& def() const { return module_def_; }
  const FunctionTable& function_table() const { return function_table_; }
  FunctionTable* mutable_function_table() { return &function_table_; }
  const ExecutableTable& executable_table() const { return executable_table_; }
  ExecutableTable* mutable_executable_table() { return &executable_table_; }

 private:
  explicit Module(std::unique_ptr<ModuleFile> module_file);

  std::unique_ptr<ModuleFile> module_file_;
  const ModuleDef& module_def_;
  FunctionTable function_table_;
  ExecutableTable executable_table_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_MODULE_H_
