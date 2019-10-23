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
#include "base/status.h"
#include "rt/stack_frame.h"

namespace iree {
namespace rt {

// A snapshot of a stack at a point in time.
// The frames within a stack may be from different backends and may provide
// varying levels of information based on capabilities.
//
// Depending on the capture options the trace may contain references to register
// values (such as buffers) from the time of capture. If the buffers were
// modified after the capture was taken those results will be reflected!
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
