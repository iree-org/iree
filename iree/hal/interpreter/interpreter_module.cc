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

#include "iree/hal/interpreter/interpreter_module.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/interpreter/bytecode_dispatch.h"
#include "iree/vm/bytecode_tables_interpreter.h"

namespace iree {
namespace hal {

// static
StatusOr<ref_ptr<rt::Module>> InterpreterModule::FromDef(
    hal::Allocator* allocator, const ModuleDef& module_def) {
  ASSIGN_OR_RETURN(auto module_file,
                   vm::ModuleFile::Create(&module_def, []() {}));
  if (module_file->root() == nullptr) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No root ModuleDef present";
  }

  auto module =
      assign_ref(new InterpreterModule(allocator, std::move(module_file)));

  // TODO(benvanik): validate internals here? or make explicit?

  return {std::move(module)};
}

InterpreterModule::InterpreterModule(
    hal::Allocator* allocator, std::unique_ptr<vm::ModuleFile> module_file)
    : vm::BytecodeModule(std::move(module_file),
                         vm::interpreter_opcode_table()),
      allocator_(allocator) {}

Status InterpreterModule::Execute(
    rt::Stack* stack, const rt::Function function,
    absl::InlinedVector<hal::BufferView, 8> arguments,
    absl::InlinedVector<hal::BufferView, 8>* results) const {
  IREE_TRACE_SCOPE0("InterperterModule::Execute");

  // Push stack frame for the function we are calling.
  ASSIGN_OR_RETURN(auto* callee_stack_frame, stack->PushFrame(function));

  // TODO(benvanik): rework register storage interface.
  ASSIGN_OR_RETURN(const auto* function_def,
                   GetFunctionDef(function.linkage(), function.ordinal()));
  auto* registers = callee_stack_frame->mutable_registers();
  registers->buffer_views.resize(function_def->bytecode()->local_count());

  // Marshal input arguments.
  for (int i = 0; i < arguments.size(); ++i) {
    registers->buffer_views[i] = std::move(arguments[i]);
  }

  // Run main dispatch loop until it exits (or errors).
  RETURN_IF_ERROR(Dispatch(allocator_, &kernel_runtime_state_, stack,
                           callee_stack_frame, results));

  // Pop the callee frame to balance out the stack.
  RETURN_IF_ERROR(stack->PopFrame());

  return OkStatus();
}

}  // namespace hal
}  // namespace iree
