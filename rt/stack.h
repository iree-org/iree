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

#ifndef IREE_RT_STACK_H_
#define IREE_RT_STACK_H_

#include <functional>

#include "absl/types/span.h"
#include "base/status.h"
#include "rt/stack_frame.h"

namespace iree {
namespace rt {

class Context;

// A runtime call stack for managing stack frames.
// The frames within a stack may be from different backends and may provide
// varying levels of information based on capabilities.
//
// Thread-compatible. Do not attempt to investigate a stack while another thread
// may be mutating it!
class Stack final {
 public:
  static constexpr int kMaxStackDepth = 32;

  explicit Stack(Context* context);
  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;
  ~Stack();

  // Context defining the module and global workspaces.
  Context* context() const { return context_; }

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

  // Returns a full stack frame listing in human-readable form.
  std::string DebugString() const;

 private:
  Context* context_ = nullptr;
  std::array<StackFrame, kMaxStackDepth> frames_;
  int stack_depth_ = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Stack& stack) {
  stream << stack.DebugString();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_STACK_H_
