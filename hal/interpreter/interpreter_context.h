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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_HAL_INTERPRETER_INTERPRETER_CONTEXT_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_HAL_INTERPRETER_INTERPRETER_CONTEXT_H_

#include <memory>

#include "third_party/absl/types/span.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/hal/allocator.h"
#include "third_party/mlir_edge/iree/hal/buffer_view.h"
#include "third_party/mlir_edge/iree/hal/interpreter/bytecode_kernels.h"
#include "third_party/mlir_edge/iree/vm/context.h"
#include "third_party/mlir_edge/iree/vm/function.h"
#include "third_party/mlir_edge/iree/vm/stack.h"

namespace iree {
namespace hal {

class InterpreterContext final : public vm::Context {
 public:
  explicit InterpreterContext(hal::Allocator* allocator)
      : allocator_(allocator) {}

  // TODO(benvanik): helpers to make passing args easier
  Status Invoke(vm::Stack* stack, vm::Function function,
                absl::Span<BufferView> args,
                absl::Span<BufferView> results) const;

 private:
  hal::Allocator* allocator_;
  mutable kernels::RuntimeState kernel_runtime_state_;
};

}  // namespace hal
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_HAL_INTERPRETER_INTERPRETER_CONTEXT_H_
