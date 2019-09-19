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

#ifndef IREE_VM_STACK_FRAME_H_
#define IREE_VM_STACK_FRAME_H_

#include <vector>

#include "absl/types/span.h"
#include "iree/hal/buffer_view.h"
#include "iree/vm/function.h"
#include "iree/vm/module.h"

namespace iree {
namespace vm {

// A single frame on the call stack containing current execution state and
// local values.
//
// StackFrames are designed to be serialized so that suspend and resume is
// possible. This means that most state is stored either entirely within the
// frame or references to non-pointer values (such as other function indices).
// BufferViews require special care to allow rendezvous and liveness tracking.
class StackFrame {
 public:
  StackFrame() = default;
  explicit StackFrame(Function function);
  explicit StackFrame(const ImportFunction& function);
  StackFrame(const StackFrame&) = delete;
  StackFrame& operator=(const StackFrame&) = delete;
  StackFrame(StackFrame&&) = default;
  StackFrame& operator=(StackFrame&&) = default;

  const Module& module() const { return function_.module(); }
  const Function& function() const { return function_; }

  inline int offset() const { return offset_; }
  Status set_offset(int offset);
  inline int* mutable_offset() { return &offset_; }

  inline const hal::BufferView& local(int ordinal) { return locals_[ordinal]; }
  inline hal::BufferView* mutable_local(int ordinal) {
    return &locals_[ordinal];
  }

  inline absl::Span<const hal::BufferView> locals() const {
    return absl::MakeConstSpan(locals_);
  }
  inline absl::Span<hal::BufferView> mutable_locals() {
    return absl::MakeSpan(locals_);
  }

 private:
  Function function_;
  const ImportFunction* import_function_;
  int offset_ = 0;
  int offset_limit_ = 0;

  // TODO(benvanik): replace with a placed allocation.
  std::vector<hal::BufferView> locals_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_STACK_FRAME_H_
