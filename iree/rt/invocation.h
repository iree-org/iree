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

#ifndef IREE_RT_INVOCATION_H_
#define IREE_RT_INVOCATION_H_

#include <ostream>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/rt/function.h"
#include "iree/rt/policy.h"
#include "iree/rt/stack.h"
#include "iree/rt/stack_trace.h"

namespace iree {
namespace rt {

class Context;

// An asynchronous invocation of a function.
// Holds the invocation state and allows querying and waiting on completion.
// Invocations are conceptually fibers and may suspend and resume execution
// several times before completing.
//
// Thread-safe.
class Invocation final : public RefObject<Invocation> {
 public:
  // TODO(benvanik): define error propagation semantics across dependencies.
  // TODO(benvanik): support more dependency types (semaphores, etc).
  // Creates a new invocation tracking object for invoking the given |function|
  // from |context|. |arguments| will be retained until the invocation is made.
  // If |dependencies| are provided then the invocation will wait until they are
  // resolved before executing. If a |policy| is provided it will override the
  // context-level policy.
  //
  // Optionally |results| may be provided with preallocated buffers that will
  // receive the outputs of the invocation. Invocation will fail if they do not
  // match expected sizes.
  //
  // Note that it's possible for the invocation to complete prior to the return
  // of this function. Any errors that occur will be set on the invocation and
  // callers should query its state prior to assuming it is in-flight.
  static StatusOr<ref_ptr<Invocation>> Create(
      ref_ptr<Context> context, const Function function, ref_ptr<Policy> policy,
      absl::InlinedVector<ref_ptr<Invocation>, 4> dependencies,
      absl::InlinedVector<hal::BufferView, 8> arguments,
      absl::optional<absl::InlinedVector<hal::BufferView, 8>> results =
          absl::nullopt);
  static StatusOr<ref_ptr<Invocation>> Create(
      ref_ptr<Context> context, const Function function, ref_ptr<Policy> policy,
      absl::Span<const ref_ptr<Invocation>> dependencies,
      absl::Span<const hal::BufferView> arguments);

  ~Invocation();

  // A process-unique ID for the invocation.
  int32_t id() const { return id_; }

  // Context this invocation is running within.
  const ref_ptr<Context>& context() const { return context_; }

  // Function being invoked.
  const Function& function() const { return function_; }

  // A single-line human-readable debug string for the invocation.
  std::string DebugStringShort() const;

  // A long-form debug string with stack trace (if available).
  std::string DebugString() const;

  // Queries the completion status of the invocation.
  // Returns one of the following:
  //   StatusCode::kOk: the invocation completed successfully.
  //   StatusCode::kUnavailable: the invocation has not yet completed.
  //   StatusCode::kCancelled: the invocation was cancelled internally.
  //   StatusCode::kAborted: the invocation was aborted.
  //   StatusCode::*: an error occurred during invocation.
  Status QueryStatus();

  // Returns ownership of the results of the operation to the caller.
  // If the invocation failed then the result will be returned as if Query had
  // been called.
  StatusOr<absl::InlinedVector<hal::BufferView, 8>> ConsumeResults();

  // Blocks the caller until the invocation completes (successfully or
  // otherwise).
  //
  // Returns StatusCode::kDeadlineExceeded if |deadline| elapses before the
  // invocation completes and otherwise returns the result of Query().
  Status Await(absl::Time deadline);

  // Attempts to abort the invocation if it is in-flight.
  // A no-op if the invocation has already completed.
  Status Abort();

  // TODO(benvanik): export a hal::TimelineSemaphore.

 private:
  friend class Context;

  Invocation(ref_ptr<Context> context, const Function function,
             ref_ptr<Policy> policy);

  // Completes the invocation with a successful result.
  void CompleteSuccess(absl::InlinedVector<hal::BufferView, 8> results);

  // Completes the invocation with a failure, including an optional stack trace.
  void CompleteFailure(Status completion_status,
                       std::unique_ptr<StackTrace> failure_stack_trace);

  int32_t id_;
  ref_ptr<Context> context_;
  const Function function_;
  ref_ptr<Policy> policy_;

  Stack stack_;

  absl::Mutex status_mutex_;
  Status completion_status_ ABSL_GUARDED_BY(status_mutex_) =
      UnavailableErrorBuilder(IREE_LOC);
  std::unique_ptr<StackTrace> failure_stack_trace_
      ABSL_GUARDED_BY(status_mutex_);
  absl::InlinedVector<hal::BufferView, 8> results_
      ABSL_GUARDED_BY(status_mutex_);

  friend class Context;
  IntrusiveListLink context_list_link_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Invocation& invocation) {
  stream << invocation.DebugStringShort();
  return stream;
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_INVOCATION_H_
