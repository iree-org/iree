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

#ifndef IREE_HAL_INTERPRETER_BYTECODE_EXECUTABLE_H_
#define IREE_HAL_INTERPRETER_BYTECODE_EXECUTABLE_H_

#include <vector>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/allocator.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_spec.h"
#include "iree/hal/interpreter/interpreter_module.h"

namespace iree {
namespace hal {

class BytecodeExecutable final : public Executable {
 public:
  static StatusOr<ref_ptr<BytecodeExecutable>> Load(hal::Allocator* allocator,
                                                    ExecutableSpec spec,
                                                    bool allow_aliasing_data);

  BytecodeExecutable(hal::Allocator* allocator, ExecutableSpec spec,
                     bool allow_aliasing_data);
  ~BytecodeExecutable() override;

  bool supports_debugging() const override { return false; }

  // Reference to the bytecode blob contents.
  absl::Span<const uint8_t> executable_data() const {
    return spec_.executable_data;
  }

  // VM module representing the executable.
  // Note that there may be more than one module in the Context and only this
  // module can be used to lookup executable exports.
  const ref_ptr<InterpreterModule>& module() const { return module_; }

 private:
  ExecutableSpec spec_;
  std::vector<uint8_t> cloned_executable_data_;

  ref_ptr<InterpreterModule> module_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_BYTECODE_EXECUTABLE_H_
