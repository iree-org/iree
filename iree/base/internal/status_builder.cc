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

#include "iree/base/internal/status_builder.h"

#include <cerrno>
#include <cstdio>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

namespace iree {

StatusBuilder::Rep::Rep(const Rep& r)
    : stream_message(r.stream_message), stream(&stream_message) {}

StatusBuilder::StatusBuilder(Status&& original_status, SourceLocation location)
    : status_(exchange(original_status, original_status.code())) {}

StatusBuilder::StatusBuilder(Status&& original_status, SourceLocation location,
                             const char* format, ...)
    : status_(exchange(original_status, original_status.code())) {
  if (status_.ok()) return;
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  status_ =
      iree_status_annotate_vf(status_.release(), format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
}

StatusBuilder::StatusBuilder(StatusCode code, SourceLocation location)
    : status_(code, location, "") {}

StatusBuilder::StatusBuilder(StatusCode code, SourceLocation location,
                             const char* format, ...) {
  if (code == StatusCode::kOk) {
    status_ = StatusCode::kOk;
    return;
  }
  va_list varargs_0, varargs_1;
  va_start(varargs_0, format);
  va_start(varargs_1, format);
  status_ = iree_status_allocate_vf(static_cast<iree_status_code_t>(code),
                                    location.file_name(), location.line(),
                                    format, varargs_0, varargs_1);
  va_end(varargs_0);
  va_end(varargs_1);
}

void StatusBuilder::Flush() {
  if (!rep_ || rep_->stream_message.empty()) return;
  auto rep = std::move(rep_);
  status_ = iree_status_annotate_f(status_.release(), "%.*s",
                                   static_cast<int>(rep->stream_message.size()),
                                   rep->stream_message.data());
}

bool StatusBuilder::ok() const { return status_.ok(); }

StatusBuilder AbortedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kAborted, location);
}

StatusBuilder AlreadyExistsErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kAlreadyExists, location);
}

StatusBuilder CancelledErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kCancelled, location);
}

StatusBuilder DataLossErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kDataLoss, location);
}

StatusBuilder DeadlineExceededErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kDeadlineExceeded, location);
}

StatusBuilder FailedPreconditionErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kFailedPrecondition, location);
}

StatusBuilder InternalErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kInternal, location);
}

StatusBuilder InvalidArgumentErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kInvalidArgument, location);
}

StatusBuilder NotFoundErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kNotFound, location);
}

StatusBuilder OutOfRangeErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kOutOfRange, location);
}

StatusBuilder PermissionDeniedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kPermissionDenied, location);
}

StatusBuilder UnauthenticatedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnauthenticated, location);
}

StatusBuilder ResourceExhaustedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kResourceExhausted, location);
}

StatusBuilder UnavailableErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnavailable, location);
}

StatusBuilder UnimplementedErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnimplemented, location);
}

StatusBuilder UnknownErrorBuilder(SourceLocation location) {
  return StatusBuilder(StatusCode::kUnknown, location);
}

StatusBuilder ErrnoToCanonicalStatusBuilder(int error_number,
                                            SourceLocation location) {
  return StatusBuilder(iree_status_code_from_errno(error_number), location);
}

#if defined(IREE_PLATFORM_WINDOWS)

StatusBuilder Win32ErrorToCanonicalStatusBuilder(uint32_t error,
                                                 SourceLocation location) {
  // TODO(benvanik): use FormatMessage; or defer until required?
  return StatusBuilder(iree_status_code_from_win32_error(error), location)
         << "<TBD>: " << error;
}

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace iree
