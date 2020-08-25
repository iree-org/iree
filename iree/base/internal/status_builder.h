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

class IREE_MUST_USE_RESULT StatusBuilder;

// Creates a status based on an original_status, but enriched with additional
// information. The builder implicitly converts to Status and StatusOr<T>
// allowing for it to be returned directly.
class StatusBuilder {
 public:
  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(Status&& original_status, SourceLocation location);
  explicit StatusBuilder(Status&& original_status, SourceLocation location,
                         const char* format, ...);

  // Creates a `StatusBuilder` from a status code.
  explicit StatusBuilder(StatusCode code, SourceLocation location);
  explicit StatusBuilder(StatusCode code, SourceLocation location,
                         const char* format, ...);

  StatusBuilder(const StatusBuilder& sb) = delete;
  StatusBuilder& operator=(const StatusBuilder& sb) = delete;
  StatusBuilder(StatusBuilder&&) noexcept;
  StatusBuilder& operator=(StatusBuilder&&) noexcept;

  // Appends to the extra message that will be added to the original status.
  template <typename T>
  StatusBuilder& operator<<(const T& value) &;
  template <typename T>
  StatusBuilder&& operator<<(const T& value) &&;

  // Returns true if the Status created by this builder will be ok().
  bool ok() const;

  StatusCode code() const { return status_.code(); }

  IREE_MUST_USE_RESULT Status ToStatus() {
    if (!status_.ok()) Flush();
    return exchange(status_, status_.code());
  }

  // Implicit conversion to Status. Eats the status object but preserves the
  // status code so the builder remains !ok().
  operator Status() && {
    if (!status_.ok()) Flush();
    return exchange(status_, status_.code());
  }

  // Implicit conversion to iree_status_t.
  operator iree_status_t() && {
    if (!status_.ok()) Flush();
    Status status = exchange(status_, status_.code());
    return static_cast<iree_status_t>(std::move(status));
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
  void Flush();

  // The status that is being built.
  // This may be an existing status that we are appending an annotation to or
  // just a code that we are building from scratch.
  Status status_;

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

inline StatusBuilder::StatusBuilder(StatusBuilder&& sb) noexcept
    : status_(exchange(sb.status_, sb.code())), rep_(std::move(sb.rep_)) {}

inline StatusBuilder& StatusBuilder::operator=(StatusBuilder&& sb) noexcept {
  status_ = exchange(sb.status_, sb.code());
  rep_ = std::move(sb.rep_);
  return *this;
}

// Disable << streaming when status messages are disabled.
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0
template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  return *this;
}
#else
template <typename T>
StatusBuilder& StatusBuilder::operator<<(const T& value) & {
  if (status_.ok()) return *this;
  if (rep_ == nullptr) rep_ = std::make_unique<Rep>();
  rep_->stream << value;
  return *this;
}
#endif  // (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) == 0

template <typename T>
StatusBuilder&& StatusBuilder::operator<<(const T& value) && {
  return std::move(operator<<(value));
}

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
#define IREE_RETURN_IF_ERROR(...)                                  \
  IREE_STATUS_IMPL_IDENTITY_(                                      \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_MACRO_)(     \
          __VA_ARGS__, IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_, \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_,              \
          IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_))               \
  (IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__),               \
   IREE_STATUS_IMPL_GET_EXPR_(__VA_ARGS__),                        \
   IREE_STATUS_IMPL_GET_ARGS_(__VA_ARGS__))

#define IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_(var, expr, ...) \
  auto var = expr;                                               \
  if (IREE_UNLIKELY(!::iree::IsOk(var)))                         \
  return ::iree::StatusBuilder(std::move(var), IREE_LOC)
#define IREE_STATUS_MACROS_IMPL_RETURN_IF_ERROR_F_(var, expr, ...) \
  auto var = expr;                                                 \
  if (IREE_UNLIKELY(!::iree::IsOk(var)))                           \
  return ::iree::StatusBuilder(std::move(var), IREE_LOC, __VA_ARGS__)

#endif  // IREE_BASE_INTERNAL_STATUS_BUILDER_H_
