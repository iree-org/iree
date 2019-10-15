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

#include "iree/vm/sequencer_module.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/buffer_view.h"
#include "iree/rt/context.h"
#include "iree/rt/instance.h"
#include "iree/vm/bytecode_tables_sequencer.h"
#include "iree/vm/sequencer_dispatch.h"

namespace iree {
namespace vm {

namespace {

using ::iree::hal::BufferView;
using ::iree::rt::Function;
using ::iree::rt::Module;

}  // namespace

// static
StatusOr<ref_ptr<rt::Module>> SequencerModule::FromDef(
    const ModuleDef& module_def) {
  ASSIGN_OR_RETURN(auto module_file, ModuleFile::Create(&module_def, []() {}));
  return FromFile(std::move(module_file));
}

// static
StatusOr<ref_ptr<rt::Module>> SequencerModule::FromFile(
    std::unique_ptr<ModuleFile> module_file) {
  if (module_file->root() == nullptr) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No root ModuleDef present";
  }
  const auto& module_def = *module_file->root();

  // Validates the structure of the module (but not bytecode).
  // This ensures we don't have flatbuffer vectors will null entries, etc.
  RETURN_IF_ERROR(BytecodeModule::ValidateStructure(module_def));

  auto module = assign_ref(new SequencerModule(std::move(module_file)));

  // TODO(benvanik): validate internals here? or make explicit?

  return {std::move(module)};
}

SequencerModule::SequencerModule(std::unique_ptr<ModuleFile> module_file)
    : BytecodeModule(std::move(module_file), sequencer_opcode_table()) {}

SequencerModule::~SequencerModule() = default;

Status SequencerModule::Execute(
    rt::Stack* stack, const Function function,
    absl::InlinedVector<hal::BufferView, 8> arguments,
    absl::InlinedVector<hal::BufferView, 8>* results) const {
  IREE_TRACE_SCOPE0("SequencerModule::Execute");

  // Push stack frame for the function we are calling.
  ASSIGN_OR_RETURN(auto* callee_stack_frame, stack->PushFrame(function));

  // TODO(benvanik): rework register storage interface.
  ASSIGN_OR_RETURN(const auto* function_def,
                   GetFunctionDef(function.linkage(), function.ordinal()));
  auto* registers = callee_stack_frame->mutable_registers();
  registers->buffer_views.resize(function_def->bytecode()->local_count());

  // Marshal input arguments.
  for (int i = 0; i < arguments.size(); ++i) {
    auto arg = arguments[i];
    auto expected_arg_type = function_def->type()->inputs()->Get(i);
    RETURN_IF_ERROR(BytecodeModule::ValidateArgType(
        arg, *expected_arg_type->type_union_as_MemRefTypeDef()))
        << "Function " << function.name() << " argument " << i;
    registers->buffer_views[i] = std::move(arg);
  }

  // TODO(benvanik): change to:
  //   get command queue (any command queue)
  //   make command buffer
  //   record dispatch
  //   submit
  //   wait on fence
  ASSIGN_OR_RETURN(
      auto placement,
      stack->context()->instance()->device_manager()->ResolvePlacement({}));
  RETURN_IF_ERROR(
      DispatchSequence(placement, stack, callee_stack_frame, results));

  // Pop the callee frame to balance out the stack.
  RETURN_IF_ERROR(stack->PopFrame());

  return OkStatus();
}

}  // namespace vm
}  // namespace iree
