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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_FIBER_STATE_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_FIBER_STATE_H_

#include <functional>

#include "third_party/absl/types/span.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/vm/instance.h"
#include "third_party/mlir_edge/iree/vm/stack.h"

namespace iree {
namespace vm {

// Fiber call stack and state machine model.
// Fibers may not line up with host application threads and execution may move
// across threads.
//
// Fibers are thread-compatible. Certain methods, such as Suspend and Resume
// (and others that explicitly call them) may be called from other threads,
// however members and other methods should be assumed safe to use only from
// the owning thread or when is_suspended returns true.
class FiberState {
 public:
  // Called when a fiber completes suspending (in response to a Suspend or Step
  // request). The |suspend_status| will indicate if the suspension was
  // successful.
  using SuspendCallback = std::function<void(Status suspend_status)>;

  struct StepTarget {
    // TODO(benvanik): step target info (matching RPC message).
    // module / function / offset
    // relative to current: once, out, return, etc
  };

  explicit FiberState(std::shared_ptr<Instance> instance);
  FiberState(const FiberState&) = delete;
  FiberState& operator=(const FiberState&) = delete;
  ~FiberState();

  // A process-unique ID for the fiber.
  int id() const { return id_; }

  const std::shared_ptr<Instance>& instance() const { return instance_; }

  // VM call stack.
  // NOTE: only valid while suspended.
  const Stack& stack() const { return stack_; }
  Stack* mutable_stack() { return &stack_; }

  // Returns true if the fiber is suspended.
  // This only returns true if the fiber has been requested to suspend with
  // Suspend and the runtime has acked the suspend. Once suspended (and until
  // resumed) fiber state will not change and may be observed from any thread.
  //
  // Safe to call from any thread.
  bool is_suspended();

  // Suspends the fiber at the next possible chance.
  //
  // Fibers have a suspension depth and each call to Suspend must be matched
  // with a call to Resume. Fibers will only resume excution when all prior
  // Suspend calls have their matching Resume called.
  //
  // Optionally callers may provide a |suspend_callback| that will be called
  // from a random thread when the fiber is suspended (or fails to suspend).
  //
  // Safe to call from any thread.
  Status Suspend(SuspendCallback suspend_callback = nullptr);

  // Resumes the fiber if it is suspended (or cancels a pending suspend).
  // This may wake threads if they are currently waiting on the fiber to
  // execute.
  //
  // Safe to call from any thread.
  Status Resume();

  // Steps fiber execution.
  // This will attempt to resume the fiber and will complete asynchronously.
  // Upon returning the fiber should be assumed resumed and callers must query
  // is_suspended to wait until the fiber suspends again. Optionally callers may
  // provide a |suspend_callback| that will be called from a random thread when
  // the fiber is suspended (or fails to suspend).
  //
  // Safe to call from any thread while the fiber is suspended.
  Status Step(StepTarget step_target,
              SuspendCallback suspend_callback = nullptr);

  std::string DebugString() const;

 private:
  std::shared_ptr<Instance> instance_;
  int id_;
  Stack stack_;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_FIBER_STATE_H_
