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

#include "absl/types/span.h"
#include "base/status.h"
#include "hal/allocator.h"
#include "hal/buffer_view.h"
#include "hal/interpreter/bytecode_kernels.h"
#include "rt/function.h"
#include "rt/module.h"
#include "rt/stack.h"
#include "vm/bytecode_module.h"
#include "vm/bytecode_tables_interpreter.h"

namespace iree {
namespace hal {

class InterpreterModule final : public vm::BytecodeModule {
 public:
  static StatusOr<ref_ptr<rt::Module>> FromDef(hal::Allocator* allocator,
                                               const ModuleDef& module_def);

  Status Execute(
      rt::Stack* stack, const rt::Function function,
      absl::InlinedVector<hal::BufferView, 8> arguments,
      absl::InlinedVector<hal::BufferView, 8>* results) const override;

 private:
  InterpreterModule(hal::Allocator* allocator,
                    std::unique_ptr<vm::ModuleFile> module_file);

  hal::Allocator* allocator_;
  mutable kernels::RuntimeState kernel_runtime_state_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_INTERPRETER_MODULE_H_
