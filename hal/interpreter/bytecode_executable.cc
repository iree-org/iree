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

#include "third_party/mlir_edge/iree/hal/interpreter/bytecode_executable.h"

#include "third_party/mlir_edge/iree/vm/bytecode_tables_interpreter.h"
#include "third_party/mlir_edge/iree/vm/bytecode_validator.h"
#include "third_party/mlir_edge/iree/vm/module.h"
#include "third_party/mlir_edge/iree/vm/module_printer.h"

namespace iree {
namespace hal {

namespace {
// TODO(benvanik): remove when debugger is wired up to the HAL.
const bool kEnableExecutablePrinting = false;
}  // namespace

// static
StatusOr<ref_ptr<BytecodeExecutable>> BytecodeExecutable::Load(
    hal::Allocator* allocator, ExecutableSpec spec, bool allow_aliasing_data) {
  // Allocate the executable now.
  // We do this here so that if we need to clone the data we are passing that
  // to the VM loader instead of the data we may not have access to later.
  auto executable =
      make_ref<BytecodeExecutable>(allocator, spec, allow_aliasing_data);
  auto* context = executable->mutable_context();

  // Create the executable module.
  auto module_def =
      ::flatbuffers::GetRoot<ModuleDef>(executable->executable_data().data());
  ASSIGN_OR_RETURN(auto module, vm::Module::FromDef(*module_def));
  executable->module_ = module.get();
  RETURN_IF_ERROR(context->RegisterModule(std::move(module)));

  // Validate bytecode to ensure it will be usable for execution.
  // We do this here so that we get a good stack immediately when the bytecode
  // is provided instead of when we go to run it. This more closely mirrors how
  // a backend that performed compilation (such as SPIR-V) would fail.
  for (auto* function_def :
       *executable->module().function_table().def().functions()) {
    RETURN_IF_ERROR(vm::BytecodeValidator::Validate(
        *context, executable->module(), *function_def->bytecode()));
  }

  // Print the bytecode.
  // TODO(benvanik): remove when debugger is wired up to the HAL.
  if (kEnableExecutablePrinting) {
    vm::PrintModuleFlagBitfield print_flags = vm::PrintModuleFlag::kNone;
    for (const auto& module : context->modules()) {
      RETURN_IF_ERROR(vm::PrintModuleToStream(
          vm::interpreter_opcode_table(), *module, print_flags, &std::cout));
    }
  }

  return executable;
}

BytecodeExecutable::BytecodeExecutable(hal::Allocator* allocator,
                                       ExecutableSpec spec,
                                       bool allow_aliasing_data)
    : spec_(spec), context_(allocator) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

BytecodeExecutable::~BytecodeExecutable() = default;

}  // namespace hal
}  // namespace iree
