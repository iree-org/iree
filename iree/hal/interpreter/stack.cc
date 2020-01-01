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

#include "iree/hal/interpreter/stack.h"

#include <iterator>

#include "iree/base/status.h"

namespace iree {
namespace hal {

constexpr int Stack::kMaxStackDepth;

Stack::Stack() = default;

Stack::~Stack() = default;

StatusOr<StackFrame*> Stack::PushFrame(Function function) {
  if (stack_depth_ + 1 > kMaxStackDepth) {
    return InternalErrorBuilder(IREE_LOC)
           << "Max stack depth of " << kMaxStackDepth << " exceeded";
  }
  frames_[stack_depth_++] = StackFrame(function);

  // TODO(benvanik): WTF scope enter.

  return current_frame();
}

Status Stack::PopFrame() {
  if (stack_depth_ == 0) {
    return InternalErrorBuilder(IREE_LOC) << "Unbalanced stack pop";
  }

  // TODO(benvanik): WTF scope leave.

  --stack_depth_;
  frames_[stack_depth_] = {};
  return OkStatus();
}

}  // namespace hal
}  // namespace iree
