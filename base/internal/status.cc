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

#include "base/internal/status.h"

#include <atomic>
#include <memory>

#include "absl/base/attributes.h"
#include "absl/debugging/stacktrace.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

ABSL_FLAG(bool, iree_status_save_stack_trace, false,
          "Save and display the full stack trace of the point of error")
    .OnUpdate([]() {
      iree::StatusSavesStackTrace(
          absl::GetFlag(FLAGS_iree_status_save_stack_trace));
    });

namespace iree {

namespace status_internal {

ABSL_CONST_INIT std::atomic<bool> iree_save_stack_trace{false};

}  // namespace status_internal

bool DoesStatusSaveStackTrace() {
  return status_internal::iree_save_stack_trace.load(std::memory_order_relaxed);
}
void StatusSavesStackTrace(bool on_off) {
  status_internal::iree_save_stack_trace.store(on_off,
                                               std::memory_order_relaxed);
}

std::string StatusCodeToString(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kCancelled:
      return "CANCELLED";
    case StatusCode::kUnknown:
      return "UNKNOWN";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case StatusCode::kDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kAlreadyExists:
      return "ALREADY_EXISTS";
    case StatusCode::kPermissionDenied:
      return "PERMISSION_DENIED";
    case StatusCode::kUnauthenticated:
      return "UNAUTHENTICATED";
    case StatusCode::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::kFailedPrecondition:
      return "FAILED_PRECONDITION";
    case StatusCode::kAborted:
      return "ABORTED";
    case StatusCode::kOutOfRange:
      return "OUT_OF_RANGE";
    case StatusCode::kUnimplemented:
      return "UNIMPLEMENTED";
    case StatusCode::kInternal:
      return "INTERNAL";
    case StatusCode::kUnavailable:
      return "UNAVAILABLE";
    case StatusCode::kDataLoss:
      return "DATA_LOSS";
    default:
      return "";
  }
}

Status::Status() {}

Status::Status(StatusCode code, absl::string_view message) {
  state_ = absl::make_unique<State>();
  state_->code = code;
  state_->message = std::string(message);
}

Status::Status(const Status& x) {
  if (x.ok()) return;

  state_ = absl::make_unique<State>();
  state_->code = x.state_->code;
  state_->message = x.state_->message;
}

Status& Status::operator=(const Status& x) {
  if (x.ok()) {
    state_ = nullptr;
  } else {
    state_ = absl::make_unique<State>();
    state_->code = x.state_->code;
    state_->message = x.state_->message;
  }
  return *this;
}

Status::~Status() {}

bool Status::ok() const { return state_ == nullptr; }

StatusCode Status::code() const {
  return ok() ? StatusCode::kOk : state_->code;
}

absl::string_view Status::message() const {
  return ok() ? absl::string_view() : absl::string_view(state_->message);
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }

  std::string text;
  absl::StrAppend(&text, StatusCodeToString(state_->code), ": ",
                  state_->message);
  // TODO(scotttodd): Payloads (stack traces)
  return text;
}

void Status::IgnoreError() const {
  // no-op
}

bool Status::EqualsSlow(const Status& a, const Status& b) {
  if (a.code() != b.code()) return false;
  if (a.message() != b.message()) return false;
  // TODO(scotttodd): Payloads
  return true;
}

bool operator==(const Status& lhs, const Status& rhs) {
  return lhs.state_ == rhs.state_ || Status::EqualsSlow(lhs, rhs);
}

bool operator!=(const Status& lhs, const Status& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

Status OkStatus() { return Status(); }

Status Annotate(const Status& s, absl::string_view msg) {
  if (s.ok() || msg.empty()) return s;

  absl::string_view new_msg = msg;
  std::string annotated;
  if (!s.message().empty()) {
    absl::StrAppend(&annotated, s.message(), "; ", msg);
    new_msg = annotated;
  }
  Status result(s.code(), new_msg);
  // TODO(scotttodd): Copy payload(s) into the new Status
  return result;
}

}  // namespace iree
