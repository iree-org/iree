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

#include "iree/base/internal/ostringstream.h"
#include "iree/base/internal/status.h"

namespace iree {

// Creates a status based on an original_status, but enriched with additional
// information. The builder implicitly converts to Status and StatusOr<T>
// allowing for it to be returned directly.
class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(const Status& original_status, SourceLocation location,
                         ...);
  explicit StatusBuilder(Status&& original_status, SourceLocation location,
                         ...);

  // Creates a `StatusBuilder` from a status code.
  // A typical user will not specify `location`, allowing it to default to the
  // current location.
  explicit StatusBuilder(StatusCode code, SourceLocation location, ...);

  explicit StatusBuilder(iree_status_t status, SourceLocation location, ...)
      : status_(static_cast<StatusCode>(iree_status_code(status)), ""),
        loc_(location) {}

  StatusBuilder(const StatusBuilder& sb);
  StatusBuilder& operator=(const StatusBuilder& sb);
  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(StatusBuilder&&) = default;

  // Appends to the extra message that will be added to the original status.
  template <typename T>
  StatusBuilder& operator<<(const T& value) &;
  template <typename T>
  StatusBuilder&& operator<<(const T& value) &&;

  // Returns true if the Status created by this builder will be ok().
  bool ok() const;

  // Returns the error code for the Status created by this builder.
  StatusCode code() const;

  // Returns the source location used to create this builder.
  SourceLocation source_location() const;

  // Implicit conversion to Status.
  operator Status() const&;
  operator Status() &&;

  // TODO(#265): toll-free result.
  operator iree_status_t() && {
    return iree_status_allocate(static_cast<iree_status_code_t>(status_.code()),
                                loc_.file_name(), loc_.line(),
                                iree_string_view_empty());
  }

  friend bool operator==(const StatusBuilder& lhs, const StatusCode& rhs) {
    return lhs.code() == rhs;
  }
  friend bool operator!=(const StatusBuilder& lhs, const StatusCode& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator==(const StatusCode& lhs, const StatusBuilder& rhs) {
    return lhs == rhs.code();
  }
  friend bool operator!=(const StatusCode& lhs, const StatusBuilder& rhs) {
    return !(lhs == rhs);
  }

 private:
  Status CreateStatus() &&;

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
    rep_ = std::make_unique<Rep>(*sb.rep_);
  }
}

inline StatusBuilder& StatusBuilder::operator=(const StatusBuilder& sb) {
  status_ = sb.status_;
  loc_ = sb.loc_;
  if (sb.rep_ != nullptr) {
    rep_ = std::make_unique<Rep>(*sb.rep_);
  } else {
    rep_ = nullptr;
  }
  return *this;
}

template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  if (status_.ok()) return *this;
  if (rep_ == nullptr) rep_ = std::make_unique<Rep>();
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

// Returns a StatusBuilder using a status code derived from |errno|.
StatusBuilder ErrnoToCanonicalStatusBuilder(int error_number,
                                            SourceLocation location);

#if defined(IREE_PLATFORM_WINDOWS)

// Returns a StatusBuilder with a status describing the |error| and |location|.
StatusBuilder Win32ErrorToCanonicalStatusBuilder(uint32_t error,
                                                 SourceLocation location);

#endif  // IREE_PLATFORM_WINDOWS

}  // namespace iree

// Override the C macro with our C++ one.
// For files that only include the C API header they'll only support
// iree_status_t results, while those including this header will support both.
// StatusBuilder can take varargs to support the printf-style formatting of the
// C macros.
#undef IREE_RETURN_IF_ERROR

// Evaluates an expression that produces a `iree::Status`. If the status is not
// ok, returns it from the current function.
#define IREE_RETURN_IF_ERROR(...)           \
  IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_( \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), __VA_ARGS__)

#define IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_(var, expr, ...) \
  auto var = (expr);                                             \
  if (IREE_UNLIKELY(!::iree::IsOk(var)))                         \
  return ::iree::StatusBuilder(std::move(var), IREE_LOC, __VA_ARGS__)

#endif  // IREE_BASE_INTERNAL_STATUS_BUILDER_H_
