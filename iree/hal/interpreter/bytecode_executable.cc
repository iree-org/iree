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

#include "iree/hal/interpreter/bytecode_executable.h"

#include <iostream>

#include "iree/hal/interpreter/interpreter_module.h"
#include "iree/rt/policy.h"

namespace iree {
namespace hal {

// static
StatusOr<ref_ptr<BytecodeExecutable>> BytecodeExecutable::Load(
    ref_ptr<rt::Instance> instance, hal::Allocator* allocator,
    ExecutableSpec spec, bool allow_aliasing_data) {
  // Allocate the executable now.
  // We do this here so that if we need to clone the data we are passing that
  // to the VM loader instead of the data we may not have access to later.
  auto executable = make_ref<BytecodeExecutable>(std::move(instance), allocator,
                                                 spec, allow_aliasing_data);

  // Create the executable module.
  auto module_def =
      ::flatbuffers::GetRoot<ModuleDef>(executable->executable_data().data());
  ASSIGN_OR_RETURN(auto module,
                   InterpreterModule::FromDef(allocator, *module_def));
  executable->module_ = add_ref(module);
  RETURN_IF_ERROR(executable->context()->RegisterModule(std::move(module)));

  return executable;
}

BytecodeExecutable::BytecodeExecutable(ref_ptr<rt::Instance> instance,
                                       hal::Allocator* allocator,
                                       ExecutableSpec spec,
                                       bool allow_aliasing_data)
    : spec_(spec),
      context_(
          make_ref<rt::Context>(std::move(instance), make_ref<rt::Policy>())) {
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
