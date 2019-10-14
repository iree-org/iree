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

#include "iree/vm/fiber_state.h"

#include <iterator>

#include "absl/strings/str_join.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

FiberState::FiberState(std::shared_ptr<Instance> instance)
    : instance_(std::move(instance)), id_(Instance::NextUniqueId()) {
}

FiberState::~FiberState() {
}

bool FiberState::is_suspended() {
  // TODO(benvanik): implement.
  return false;
}

Status FiberState::Suspend(SuspendCallback suspend_callback) {
  DVLOG(1) << "Suspending fiber " << id();
  return OkStatus();
}

Status FiberState::Resume() {
  DVLOG(1) << "Resuming fiber " << id();
  return OkStatus();
}

Status FiberState::Step(StepTarget step_target,
                        SuspendCallback suspend_callback) {
  return UnimplementedErrorBuilder(IREE_LOC) << "Step not yet implemented";
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

std::string FiberState::DebugString() const {
  auto frames = stack_.frames();
  return absl::StrJoin(frames.begin(), frames.end(), "\n",
                       StackFrameFormatter());
}

}  // namespace vm
}  // namespace iree
