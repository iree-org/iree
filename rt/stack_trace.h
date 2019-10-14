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

#ifndef IREE_RT_STACK_TRACE_H_
#define IREE_RT_STACK_TRACE_H_

#include <ostream>
#include <vector>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/rt/stack_frame.h"

namespace iree {
namespace rt {

// A snapshot of the runtime callstack providing access to stack frames.
// The frames within a stack may be from different backends and may provide
// varying levels of information based on capabilities.
//
// Thread-compatible. Execution on one thread and stack manipulation on another
// must be externally synchronized by the caller.
class StackTrace final {
 public:
  StackTrace() = default;
  explicit StackTrace(std::vector<StackFrame> frames)
      : frames_(std::move(frames)) {}
  StackTrace(const StackTrace&) = delete;
  StackTrace& operator=(const StackTrace&) = delete;
  ~StackTrace() = default;

  // All stack frames within the stack.
  absl::Span<const StackFrame> frames() const {
    return absl::MakeConstSpan(frames_);
  }

  // The current stack frame.
  const StackFrame* current_frame() const {
    return !frames_.empty() ? &frames_[frames_.size() - 1] : nullptr;
  }

  // The stack frame of the caller of the current function.
  const StackFrame* caller_frame() const {
    return frames_.size() > 1 ? &frames_[frames_.size() - 2] : nullptr;
  }

  // Returns a full stack frame listing in human-readable form.
  std::string DebugString() const;

 private:
  std::vector<StackFrame> frames_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const StackTrace& stack_trace) {
  stream << stack_trace.DebugString();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_STACK_TRACE_H_
