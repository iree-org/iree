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

#ifndef IREE_HAL_INTERPRETER_INTERPRETER_MODULE_H_
#define IREE_HAL_INTERPRETER_INTERPRETER_MODULE_H_

#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/interpreter/bytecode_kernels.h"
#include "iree/hal/interpreter/bytecode_tables_interpreter.h"
#include "iree/schemas/interpreter_module_def_generated.h"

namespace iree {
namespace hal {

class InterpreterModule;
class Stack;

using ModuleFile = FlatBufferFile<ModuleDef>;

// Reference to a function within a module.
// Functions are either visible or hidden from the module interface and may be
// of one Linkage type. Imports and exports are always visible (as they are
// required for dynamic linking) however functions with internal linkage may be
// hidden in optimized builds to reduce the amount of reflection metadata
// required.
class Function final {
 public:
  enum class Linkage {
    // Function is internal to the module and may not be reflectable.
    kInternal = 0,
    // Function is an import from another module.
    kImport = 1,
    // Function is an export from the module.
    kExport = 2,
  };

  Function() = default;
  Function(const InterpreterModule* module, Linkage linkage, int32_t ordinal)
      : module_(module), linkage_(linkage), ordinal_(ordinal) {}

  // Module the function is contained within.
  const InterpreterModule* module() const { return module_; }

  // Linkage of the function. Note that Linkage::kInternal functions may be
  // missing reflection information.
  Linkage linkage() const { return linkage_; }

  // Ordinal within the module in the linkage scope.
  int32_t ordinal() const { return ordinal_; }

 private:
  const InterpreterModule* module_ = nullptr;
  Linkage linkage_ = Linkage::kInternal;
  int32_t ordinal_ = -1;
};

class InterpreterModule final : public RefObject<InterpreterModule> {
 public:
  static Status ValidateStructure(const ModuleDef& module_def);

  static StatusOr<ref_ptr<InterpreterModule>> FromDef(
      hal::Allocator* allocator, const ModuleDef& module_def);

  const ModuleDef& def() const { return module_def_; }
  const FunctionTableDef& function_table_def() const {
    return *module_def_.function_table();
  }

  // Looks up a visible function by ordinal.
  // Internal functions may not be found if debugging info has been stripped.
  StatusOr<const Function> LookupFunctionByOrdinal(Function::Linkage linkage,
                                                   int32_t ordinal) const;

  StatusOr<const FunctionDef*> GetFunctionDef(Function::Linkage linkage,
                                              int32_t ordinal) const;

  Status Execute(Stack* stack, const Function function,
                 absl::InlinedVector<hal::BufferView, 8> arguments,
                 absl::InlinedVector<hal::BufferView, 8>* results) const;

 private:
  static Status ValidateArgType(const hal::BufferView& arg,
                                const MemRefTypeDef& expected_type);

  InterpreterModule(hal::Allocator* allocator, ref_ptr<ModuleFile> module_file);

  StatusOr<int32_t> MapFunctionOrdinal(Function::Linkage linkage,
                                       int32_t ordinal) const;

  hal::Allocator* allocator_;
  mutable kernels::RuntimeState kernel_runtime_state_;
  ref_ptr<ModuleFile> module_file_;
  const ModuleDef& module_def_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_INTERPRETER_MODULE_H_
