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

#ifndef IREE_RT_STACK_FRAME_H_
#define IREE_RT_STACK_FRAME_H_

#include <ostream>

#include "absl/types/span.h"
#include "rt/function.h"
#include "rt/module.h"
#include "rt/source_location.h"

namespace iree {
namespace rt {

// TODO(benvanik): allocate in-place from an arena.
// Register table used within a stack frame.
struct Registers {
  std::vector<hal::BufferView> buffer_views;
};

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
  const Module& module() const { return *function_.module(); }

  // Function the stack frame represents.
  const Function& function() const { return function_; }

  // Current virtual offset within the function.
  // The exact meaning of the offset is backend dependent and callers should
  // treat them as opaque and must use the SourceResolver to compute new
  // offsets (such as 'next offset').
  SourceOffset offset() const { return offset_; }
  SourceOffset* mutable_offset() { return &offset_; }

  // Returns a source location, if available, for the current offset within the
  // target function.
  absl::optional<SourceLocation> source_location() const;

  // Registers used within the stack frame.
  // Storage is implementation-defined and is valid only for the lifetime of the
  // frame.
  const Registers& registers() const { return registers_; }
  Registers* mutable_registers() { return &registers_; }

  // A short human-readable string for the frame; a single line.
  std::string DebugStringShort() const;

 private:
  Function function_;
  SourceOffset offset_ = 0;
  Registers registers_;
};

struct StackFrameFormatter {
  void operator()(std::string* out, const StackFrame& stack_frame) const {
    out->append(stack_frame.DebugStringShort());
  }
};

inline std::ostream& operator<<(std::ostream& stream,
                                const StackFrame& stack_frame) {
  stream << stack_frame.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_STACK_FRAME_H_
