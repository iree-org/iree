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

#include "iree/vm/stack.h"

#include <iterator>

#include "absl/strings/str_join.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

constexpr int Stack::kMaxStackDepth;

Stack::Stack() = default;

Stack::~Stack() = default;

StatusOr<StackFrame*> Stack::PushFrame(Function function) {
  if (stack_depth_ + 1 > kMaxStackDepth) {
    return InternalErrorBuilder(ABSL_LOC)
           << "Max stack depth of " << kMaxStackDepth << " exceeded";
  }
  stack_[stack_depth_++] = StackFrame(function);

  // TODO(benvanik): WTF scope enter.

  return current_frame();
}

StatusOr<StackFrame*> Stack::PushFrame(const ImportFunction& function) {
  if (stack_depth_ + 1 > kMaxStackDepth) {
    return InternalErrorBuilder(ABSL_LOC)
           << "Max stack depth of " << kMaxStackDepth << " exceeded";
  }
  stack_[stack_depth_++] = StackFrame(function);

  // TODO(benvanik): WTF scope enter.

  return current_frame();
}

Status Stack::PopFrame() {
  if (stack_depth_ == 0) {
    return InternalErrorBuilder(ABSL_LOC) << "Unbalanced stack pop";
  }

  // TODO(benvanik): WTF scope leave.

  --stack_depth_;
  return OkStatus();
}

namespace {
struct StackFrameFormatter {
  void operator()(std::string* out, const StackFrame& stack_frame) const {
    out->append(absl::StrCat(stack_frame.module().name(), ":",
                             stack_frame.function().name(), "@",
                             stack_frame.offset()));
  }
};
}  // namespace

std::string Stack::DebugString() const {
  return absl::StrJoin(std::begin(stack_), std::begin(stack_) + stack_depth_,
                       "\n", StackFrameFormatter());
}

}  // namespace vm
}  // namespace iree
