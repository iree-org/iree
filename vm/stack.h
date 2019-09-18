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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_STACK_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_STACK_H_

#include <functional>

#include "third_party/absl/types/span.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/vm/stack_frame.h"

namespace iree {
namespace vm {

// VM call stack.
//
// Stacks are thread-compatible.
class Stack {
 public:
  static constexpr int kMaxStackDepth = 32;

  Stack();
  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;
  ~Stack();

  absl::Span<const StackFrame> frames() const {
    return absl::MakeConstSpan(stack_, stack_depth_);
  }
  absl::Span<StackFrame> mutable_frames() {
    return absl::MakeSpan(stack_, stack_depth_);
  }

  StackFrame* current_frame() {
    return stack_depth_ > 0 ? &stack_[stack_depth_ - 1] : nullptr;
  }
  const StackFrame* current_frame() const {
    return stack_depth_ > 0 ? &stack_[stack_depth_ - 1] : nullptr;
  }
  StackFrame* caller_frame() {
    return stack_depth_ > 1 ? &stack_[stack_depth_ - 2] : nullptr;
  }
  const StackFrame* caller_frame() const {
    return stack_depth_ > 1 ? &stack_[stack_depth_ - 2] : nullptr;
  }

  StatusOr<StackFrame*> PushFrame(Function function);
  StatusOr<StackFrame*> PushFrame(const ImportFunction& function);
  Status PopFrame();

  std::string DebugString() const;

 private:
  StackFrame stack_[kMaxStackDepth];
  int stack_depth_ = 0;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_STACK_H_
