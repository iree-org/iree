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

#ifndef IREE_RT_DEBUG_ADAPTER_H_
#define IREE_RT_DEBUG_ADAPTER_H_

#include <functional>

#include "iree/base/status.h"
#include "iree/rt/invocation.h"

namespace iree {
namespace rt {
namespace debug {

struct StepTarget {
  // TODO(benvanik): step target info (matching RPC message).
  // module / function / offset
  // relative to current: once, out, return, etc
};

// TODO(benvanik): move to fiber base.
// Interface for debugging invocations.
// This is only accessible in debug builds where such features are compiled in.
class DebugAdapter {
 public:
  // Called when an invocation completes suspending (in response to a Suspend or
  // Step request). The |suspend_status| will indicate if the suspension was
  // successful.
  using SuspendCallback = std::function<void(Status suspend_status)>;

  // Returns true if the invocation is suspended.
  // This only returns true if the invocation has been requested to suspend with
  // Suspend and the runtime has acked the suspend. Once suspended (and until
  // resumed) invocation state will not change and may be observed from any
  // thread.
  //
  // Safe to call from any thread.
  bool IsSuspended(Invocation* invocation);

  // Suspends the invocation at the next possible chance.
  //
  // Fibers have a suspension depth and each call to Suspend must be matched
  // with a call to Resume. Fibers will only resume excution when all prior
  // Suspend calls have their matching Resume called.
  //
  // Optionally callers may provide a |suspend_callback| that will be called
  // from a random thread when the invocation is suspended (or fails to
  // suspend).
  //
  // Safe to call from any thread.
  // Returns StatusCode::kUnavailable if debugging is not supported.
  Status Suspend(ref_ptr<Invocation> invocation,
                 SuspendCallback suspend_callback = nullptr);

  // Resumes the invocation if it is suspended (or cancels a pending suspend).
  // This may wake threads if they are currently waiting on the invocation to
  // execute.
  //
  // Safe to call from any thread.
  // Returns StatusCode::kUnavailable if debugging is not supported.
  Status Resume(Invocation* invocation);

  // Steps invocation execution.
  // This will attempt to resume the invocation and will complete
  // asynchronously. Upon returning the invocation should be assumed resumed and
  // callers must query is_suspended to wait until the invocation suspends
  // again. Optionally callers may provide a |suspend_callback| that will be
  // called from a random thread when the invocation is suspended (or fails to
  // suspend).
  //
  // Safe to call from any thread while the invocation is suspended.
  // Returns StatusCode::kUnavailable if debugging is not supported and
  // StatusCode::kFailedPrecondition if the invocation is not suspended.
  Status Step(ref_ptr<Invocation> invocation, StepTarget step_target,
              SuspendCallback suspend_callback = nullptr);

  // Returns a call stack that can be used to query and manipulate the
  // invocation state. The behaviors supported depend on the stack frames and
  // the backend support and may be conditionally enabled via compile-time or
  // run-time flags.
  //
  // Safe to call from any thread while the invocation is suspended.
  // Returns StatusCode::kUnavailable if debugging is not supported and
  // StatusCode::kFailedPrecondition if the invocation is not suspended.
  // The returned stack will be invalidated when the invocation is stepped or
  // resumed.
  StatusOr<Stack> CaptureStack(Invocation* invocation);
};

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DEBUG_ADAPTER_H_
