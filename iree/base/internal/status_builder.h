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

#ifndef IREE_BASE_INTERNAL_STATUS_BUILDER_H_
#define IREE_BASE_INTERNAL_STATUS_BUILDER_H_

#include "iree/base/internal/status.h"
#include "iree/base/source_location.h"

namespace iree {

// Creates a status based on an original_status, but enriched with additional
// information. The builder implicitly converts to Status and StatusOr<T>
// allowing for it to be returned directly.
class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(const Status& original_status,
                         SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
  explicit StatusBuilder(Status&& original_status,
                         SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);

  // Creates a `StatusBuilder` from a status code.
  // A typical user will not specify `location`, allowing it to default to the
  // current location.
  explicit StatusBuilder(StatusCode code,
                         SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);

  StatusBuilder(const StatusBuilder& sb);
  StatusBuilder& operator=(const StatusBuilder& sb);
  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(StatusBuilder&&) = default;

  // Appends to the extra message that will be added to the original status.
  template <typename T>
  StatusBuilder& operator<<(const T& value) &;
  template <typename T>
  StatusBuilder&& operator<<(const T& value) &&;

  // No-op functions that may be added later.
  StatusBuilder& LogError() & { return *this; }
  StatusBuilder&& LogError() && { return std::move(LogError()); }
  StatusBuilder& LogWarning() & { return *this; }
  StatusBuilder&& LogWarning() && { return std::move(LogWarning()); }
  StatusBuilder& LogInfo() & { return *this; }
  StatusBuilder&& LogInfo() && { return std::move(LogInfo()); }

  // Returns true if the Status created by this builder will be ok().
  bool ok() const;

  // Returns the error code for the Status created by this builder.
  StatusCode code() const;

  // Returns the source location used to create this builder.
  SourceLocation source_location() const;

  // Implicit conversion to Status.
  operator Status() const&;
  operator Status() &&;

 private:
  Status CreateStatus() &&;

  static Status JoinMessageToStatus(Status s, absl::string_view msg);

  // The status that the result will be based on.
  Status status_;

  // The location to record if this status is logged.
  SourceLocation loc_;

  // The message that will be added to the original status.
  std::string message_;
};

template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  return *this;
}
template <typename T>
StatusBuilder&& StatusBuilder::operator<<(const T& value) && {
  return std::move(operator<<(value));
}

// Implicitly converts `builder` to `Status` and write it to `os`.
std::ostream& operator<<(std::ostream& os, const StatusBuilder& builder);
std::ostream& operator<<(std::ostream& os, StatusBuilder&& builder);

// Each of the functions below creates StatusBuilder with a canonical error.
// The error code of the StatusBuilder matches the name of the function.
StatusBuilder AbortedErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder AlreadyExistsErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder CancelledErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder DataLossErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder DeadlineExceededErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder FailedPreconditionErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder InternalErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder InvalidArgumentErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder NotFoundErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder OutOfRangeErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder PermissionDeniedErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder UnauthenticatedErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder ResourceExhaustedErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder UnavailableErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder UnimplementedErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);
StatusBuilder UnknownErrorBuilder(
    SourceLocation location IREE_LOC_CURRENT_DEFAULT_ARG);

}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_BUILDER_H_
