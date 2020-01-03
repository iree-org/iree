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

#ifndef IREE_HAL_INTERPRETER_STACK_H_
#define IREE_HAL_INTERPRETER_STACK_H_

#include <functional>
#include <vector>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/interpreter/interpreter_module.h"

namespace iree {
namespace hal {

// TODO(benvanik): allocate in-place from an arena.
// Register table used within a stack frame.
struct Registers {
  std::vector<hal::BufferView> buffer_views;
};

using SourceOffset = uint64_t;

// A single frame on the call stack containing current execution state and
// register values.
//
// As different backends may support different features this interface exposes
// only the things we want to view in our debugger/stack dumps. This allows us
// to ignore the actual implementation (bytecode VM, compiled C code, etc) so
// long as it can respond to queries for register values. This has the benefit
// of keeping the actual frame very lightweight as we are not storing the values
// but instead just routing to the real storage via indirection. If the debugger
// is not attached and no errors are hit then no additional bookkeeping is done.
//
// Thread-compatible, as is the owning Stack/StackTrace.
class StackFrame final {
 public:
  StackFrame() = default;
  explicit StackFrame(Function function) : function_(function) {}
  StackFrame(Function function, SourceOffset offset, Registers registers)
      : function_(function),
        offset_(offset),
        registers_(std::move(registers)) {}
  StackFrame(const StackFrame&) = delete;
  StackFrame& operator=(const StackFrame&) = delete;
  StackFrame(StackFrame&&) = default;
  StackFrame& operator=(StackFrame&&) = default;

  // Module that owns the function this stack frame represents.
  const InterpreterModule& module() const { return *function_.module(); }

  // Function the stack frame represents.
  const Function& function() const { return function_; }

  // Current virtual offset within the function.
  // The exact meaning of the offset is backend dependent and callers should
  // treat them as opaque and must use the SourceResolver to compute new
  // offsets (such as 'next offset').
  SourceOffset offset() const { return offset_; }
  SourceOffset* mutable_offset() { return &offset_; }

  // Registers used within the stack frame.
  // Storage is implementation-defined and is valid only for the lifetime of the
  // frame.
  const Registers& registers() const { return registers_; }
  Registers* mutable_registers() { return &registers_; }

 private:
  Function function_;
  SourceOffset offset_ = 0;
  Registers registers_;
};

// A runtime call stack for managing stack frames.
// The frames within a stack may be from different backends and may provide
// varying levels of information based on capabilities.
//
// Thread-compatible. Do not attempt to investigate a stack while another thread
// may be mutating it!
class Stack final {
 public:
  static constexpr int kMaxStackDepth = 32;

  Stack();
  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;
  ~Stack();

  // All stack frames within the stack.
  absl::Span<StackFrame> frames() {
    return absl::MakeSpan(frames_).subspan(0, stack_depth_);
  }
  absl::Span<const StackFrame> frames() const {
    return absl::MakeConstSpan(frames_).subspan(0, stack_depth_);
  }

  // The current stack frame.
  StackFrame* current_frame() {
    return stack_depth_ > 0 ? &frames_[stack_depth_ - 1] : nullptr;
  }

  // The stack frame of the caller of the current function.
  StackFrame* caller_frame() {
    return stack_depth_ > 1 ? &frames_[stack_depth_ - 2] : nullptr;
  }

  StatusOr<StackFrame*> PushFrame(Function function);
  Status PopFrame();

 private:
  std::array<StackFrame, kMaxStackDepth> frames_;
  int stack_depth_ = 0;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_STACK_H_
