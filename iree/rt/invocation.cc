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

#include "iree/rt/invocation.h"

#include <atomic>
#include <iterator>

#include "absl/strings/str_cat.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/rt/context.h"

namespace iree {
namespace rt {

namespace {

int32_t NextUniqueInvocationId() {
  static std::atomic<int32_t> next_id = {0};
  return ++next_id;
}

}  // namespace

// static
StatusOr<ref_ptr<Invocation>> Invocation::Create(
    ref_ptr<Context> context, const Function function, ref_ptr<Policy> policy,
    absl::InlinedVector<ref_ptr<Invocation>, 4> dependencies,
    absl::InlinedVector<hal::BufferView, 8> arguments,
    absl::optional<absl::InlinedVector<hal::BufferView, 8>> results) {
  IREE_TRACE_SCOPE0("Invocation::Create");

  const auto& signature = function.signature();
  if (arguments.size() != signature.argument_count()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Argument count mismatch; expected " << signature.argument_count()
           << " but received " << arguments.size();
  } else if (results.has_value() &&
             results.value().size() != signature.result_count()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Result count mismatch; expected " << signature.result_count()
           << " but received " << results.value().size();
  }

  absl::InlinedVector<hal::BufferView, 8> results_value;
  if (results.has_value()) {
    results_value = std::move(results.value());
  } else {
    results_value.resize(signature.result_count());
  }

  auto invocation = assign_ref(
      new Invocation(std::move(context), function, std::move(policy)));

  // TODO(benvanik): grab execution state, insert deps, etc.
  if (!dependencies.empty()) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "Dependencies are not yet implemented";
  }

  // TODO(benvanik): fiber scheduling and such.
  auto execute_status = function.module()->Execute(
      &invocation->stack_, function, std::move(arguments), &results_value);
  if (execute_status.ok()) {
    invocation->CompleteSuccess(std::move(results_value));
  } else {
    invocation->CompleteFailure(std::move(execute_status), nullptr);
  }

  return invocation;
}

// static
StatusOr<ref_ptr<Invocation>> Invocation::Create(
    ref_ptr<Context> context, const Function function, ref_ptr<Policy> policy,
    absl::Span<const ref_ptr<Invocation>> dependencies,
    absl::Span<const hal::BufferView> arguments) {
  absl::InlinedVector<ref_ptr<Invocation>, 4> dependency_list;
  dependency_list.reserve(dependencies.size());
  for (auto& dependency : dependencies) {
    dependency_list.push_back(add_ref(dependency));
  }
  absl::InlinedVector<hal::BufferView, 8> argument_list;
  argument_list.reserve(arguments.size());
  for (auto& buffer_view : arguments) {
    argument_list.push_back(buffer_view);
  }
  return Invocation::Create(std::move(context), function, std::move(policy),
                            std::move(dependency_list),
                            std::move(argument_list));
}

Invocation::Invocation(ref_ptr<Context> context, const Function function,
                       ref_ptr<Policy> policy)
    : id_(NextUniqueInvocationId()),
      context_(std::move(context)),
      function_(function),
      policy_(std::move(policy)),
      stack_(context_.get()) {
  IREE_TRACE_SCOPE0("Invocation::ctor");
  context_->RegisterInvocation(this);
}

Invocation::~Invocation() {
  IREE_TRACE_SCOPE0("Invocation::dtor");
  context_->UnregisterInvocation(this);
}

std::string Invocation::DebugStringShort() const {
  return absl::StrCat("invocation_", id_);
}

std::string Invocation::DebugString() const { return DebugStringShort(); }

Status Invocation::QueryStatus() {
  IREE_TRACE_SCOPE0("Invocation::QueryStatus");
  absl::MutexLock lock(&status_mutex_);
  return completion_status_;
}

StatusOr<absl::InlinedVector<hal::BufferView, 8>> Invocation::ConsumeResults() {
  IREE_TRACE_SCOPE0("Invocation::ConsumeResults");
  absl::MutexLock lock(&status_mutex_);
  if (!completion_status_.ok()) {
    return completion_status_;
  }
  return std::move(results_);
}

Status Invocation::Await(absl::Time deadline) {
  IREE_TRACE_SCOPE0("Invocation::Await");
  absl::MutexLock lock(&status_mutex_);
  // TODO(benvanik): implement async invocation behavior.
  return completion_status_;
}

Status Invocation::Abort() {
  IREE_TRACE_SCOPE0("Invocation::Abort");
  // TODO(benvanik): implement async invocation behavior.
  return UnimplementedErrorBuilder(IREE_LOC)
         << "Async invocations not yet implemented";
}

void Invocation::CompleteSuccess(
    absl::InlinedVector<hal::BufferView, 8> results) {
  IREE_TRACE_SCOPE0("Invocation::CompleteSuccess");
  absl::MutexLock lock(&status_mutex_);
  if (IsAborted(completion_status_)) {
    // Ignore as the invocation was already aborted prior to completion.
    return;
  }
  DCHECK(IsUnavailable(completion_status_));
  completion_status_ = OkStatus();
  failure_stack_trace_.reset();
  results_ = std::move(results);
}

void Invocation::CompleteFailure(
    Status completion_status, std::unique_ptr<StackTrace> failure_stack_trace) {
  IREE_TRACE_SCOPE0("Invocation::CompleteFailure");
  absl::MutexLock lock(&status_mutex_);
  if (IsAborted(completion_status_)) {
    // Ignore as the invocation was already aborted prior to completion.
    return;
  }
  DCHECK(IsUnavailable(completion_status_));
  completion_status_ = std::move(completion_status);
  failure_stack_trace_ = std::move(failure_stack_trace);
  results_.clear();
}

}  // namespace rt
}  // namespace iree
