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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/internal/ostringstream.h"
#include "iree/base/internal/status.h"

namespace iree {

// Creates a status based on an original_status, but enriched with additional
// information. The builder implicitly converts to Status and StatusOr<T>
// allowing for it to be returned directly.
class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(const Status& original_status,
                         SourceLocation location);
  explicit StatusBuilder(Status&& original_status, SourceLocation location);

  // Creates a `StatusBuilder` from a status code.
  // A typical user will not specify `location`, allowing it to default to the
  // current location.
  explicit StatusBuilder(StatusCode code, SourceLocation location);

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

  // Lazy construction of the expensive stream.
  struct Rep {
    explicit Rep() = default;
    Rep(const Rep& r);

    // Gathers additional messages added with `<<` for use in the final status.
    std::string stream_message;
    iree::OStringStream stream{&stream_message};
  };

  std::unique_ptr<Rep> rep_;
};

inline StatusBuilder::StatusBuilder(const StatusBuilder& sb)
    : status_(sb.status_), loc_(sb.loc_) {
  if (sb.rep_ != nullptr) {
    rep_ = absl::make_unique<Rep>(*sb.rep_);
  }
}

inline StatusBuilder& StatusBuilder::operator=(const StatusBuilder& sb) {
  status_ = sb.status_;
  loc_ = sb.loc_;
  if (sb.rep_ != nullptr) {
    rep_ = absl::make_unique<Rep>(*sb.rep_);
  } else {
    rep_ = nullptr;
  }
  return *this;
}

template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  if (status_.ok()) return *this;
  if (rep_ == nullptr) rep_ = absl::make_unique<Rep>();
  rep_->stream << value;
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
StatusBuilder AbortedErrorBuilder(SourceLocation location);
StatusBuilder AlreadyExistsErrorBuilder(SourceLocation location);
StatusBuilder CancelledErrorBuilder(SourceLocation location);
StatusBuilder DataLossErrorBuilder(SourceLocation location);
StatusBuilder DeadlineExceededErrorBuilder(SourceLocation location);
StatusBuilder FailedPreconditionErrorBuilder(SourceLocation location);
StatusBuilder InternalErrorBuilder(SourceLocation location);
StatusBuilder InvalidArgumentErrorBuilder(SourceLocation location);
StatusBuilder NotFoundErrorBuilder(SourceLocation location);
StatusBuilder OutOfRangeErrorBuilder(SourceLocation location);
StatusBuilder PermissionDeniedErrorBuilder(SourceLocation location);
StatusBuilder UnauthenticatedErrorBuilder(SourceLocation location);
StatusBuilder ResourceExhaustedErrorBuilder(SourceLocation location);
StatusBuilder UnavailableErrorBuilder(SourceLocation location);
StatusBuilder UnimplementedErrorBuilder(SourceLocation location);
StatusBuilder UnknownErrorBuilder(SourceLocation location);

}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_BUILDER_H_
