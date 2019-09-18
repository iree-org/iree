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

#include "third_party/mlir_edge/iree/base/internal/status_builder.h"

#include <cstdio>

#include "third_party/mlir_edge/iree/base/internal/status_errors.h"

namespace iree {

StatusBuilder::StatusBuilder(const Status& original_status,
                             absl::SourceLocation location)
    : status_(original_status), loc_(location) {}

StatusBuilder::StatusBuilder(Status&& original_status,
                             absl::SourceLocation location)
    : status_(original_status), loc_(location) {}

StatusBuilder::StatusBuilder(const StatusBuilder& sb)
    : status_(sb.status_), loc_(sb.loc_), message_(sb.message_) {}

StatusBuilder::StatusBuilder(StatusCode code, absl::SourceLocation location)
    : status_(code, ""), loc_(location) {}

StatusBuilder& StatusBuilder::operator=(const StatusBuilder& sb) {
  status_ = sb.status_;
  loc_ = sb.loc_;
  message_ = sb.message_;
  return *this;
}

StatusBuilder::operator Status() const& {
  return StatusBuilder(*this).CreateStatus();
}
StatusBuilder::operator Status() && { return std::move(*this).CreateStatus(); }

bool StatusBuilder::ok() const { return status_.ok(); }

StatusCode StatusBuilder::code() const { return status_.code(); }

absl::SourceLocation StatusBuilder::source_location() const { return loc_; }

Status StatusBuilder::CreateStatus() && {
  Status result = JoinMessageToStatus(status_, message_);

  // Reset the status after consuming it.
  status_ = UnknownError("");
  message_ = "";
  return result;
}

Status StatusBuilder::JoinMessageToStatus(Status s, absl::string_view msg) {
  if (msg.empty()) return s;
  return Annotate(s, msg);
}

std::ostream& operator<<(std::ostream& os, const StatusBuilder& builder) {
  return os << static_cast<Status>(builder);
}

std::ostream& operator<<(std::ostream& os, StatusBuilder&& builder) {
  return os << static_cast<Status>(std::move(builder));
}

StatusBuilder AbortedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kAborted, location);
}

StatusBuilder AlreadyExistsErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kAlreadyExists, location);
}

StatusBuilder CancelledErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kCancelled, location);
}

StatusBuilder DataLossErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kDataLoss, location);
}

StatusBuilder DeadlineExceededErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kDeadlineExceeded, location);
}

StatusBuilder FailedPreconditionErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kFailedPrecondition, location);
}

StatusBuilder InternalErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kInternal, location);
}

StatusBuilder InvalidArgumentErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kInvalidArgument, location);
}

StatusBuilder NotFoundErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kNotFound, location);
}

StatusBuilder OutOfRangeErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kOutOfRange, location);
}

StatusBuilder PermissionDeniedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kPermissionDenied, location);
}

StatusBuilder UnauthenticatedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kUnauthenticated, location);
}

StatusBuilder ResourceExhaustedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kResourceExhausted, location);
}

StatusBuilder UnavailableErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kUnavailable, location);
}

StatusBuilder UnimplementedErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kUnimplemented, location);
}

StatusBuilder UnknownErrorBuilder(absl::SourceLocation location) {
  return StatusBuilder(StatusCode::kUnknown, location);
}

}  // namespace iree
