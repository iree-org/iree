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

#ifndef IREE_BASE_INTERNAL_STATUS_H_
#define IREE_BASE_INTERNAL_STATUS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/base/target_platform.h"

namespace iree {

// Class representing a specific location in the source code of a program.
class SourceLocation {
 public:
  // Avoid this constructor; it populates the object with dummy values.
  constexpr SourceLocation() : line_(0), file_name_(nullptr) {}

  // `file_name` must outlive all copies of the `iree::SourceLocation` object,
  // so in practice it should be a string literal.
  constexpr SourceLocation(std::uint_least32_t line, const char* file_name)
      : line_(line), file_name_(file_name) {}

  // The line number of the captured source location.
  constexpr std::uint_least32_t line() const { return line_; }

  // The file name of the captured source location.
  constexpr const char* file_name() const { return file_name_; }

 private:
  std::uint_least32_t line_;
  const char* file_name_;
};

// If a function takes an `iree::SourceLocation` parameter, pass this as the
// argument.
#if IREE_STATUS_FEATURES == 0
#define IREE_LOC ::iree::SourceLocation(0, NULL)
#else
#define IREE_LOC ::iree::SourceLocation(__LINE__, __FILE__)
#endif  // IREE_STATUS_FEATURES == 0

enum class StatusCode : uint32_t {
  kOk = IREE_STATUS_OK,
  kCancelled = IREE_STATUS_CANCELLED,
  kUnknown = IREE_STATUS_UNKNOWN,
  kInvalidArgument = IREE_STATUS_INVALID_ARGUMENT,
  kDeadlineExceeded = IREE_STATUS_DEADLINE_EXCEEDED,
  kNotFound = IREE_STATUS_NOT_FOUND,
  kAlreadyExists = IREE_STATUS_ALREADY_EXISTS,
  kPermissionDenied = IREE_STATUS_PERMISSION_DENIED,
  kResourceExhausted = IREE_STATUS_RESOURCE_EXHAUSTED,
  kFailedPrecondition = IREE_STATUS_FAILED_PRECONDITION,
  kAborted = IREE_STATUS_ABORTED,
  kOutOfRange = IREE_STATUS_OUT_OF_RANGE,
  kUnimplemented = IREE_STATUS_UNIMPLEMENTED,
  kInternal = IREE_STATUS_INTERNAL,
  kUnavailable = IREE_STATUS_UNAVAILABLE,
  kDataLoss = IREE_STATUS_DATA_LOSS,
  kUnauthenticated = IREE_STATUS_UNAUTHENTICATED,
};

static inline const char* StatusCodeToString(StatusCode code) {
  return iree_status_code_string(static_cast<iree_status_code_t>(code));
}

// Prints a human-readable representation of `x` to `os`.
std::ostream& operator<<(std::ostream& os, const StatusCode& x);

class IREE_MUST_USE_RESULT Status;

// A Status value can be either OK or not-OK
//   * OK indicates that the operation succeeded.
//   * A not-OK value indicates that the operation failed and contains details
//     about the error.
class Status final {
 public:
  // Creates an OK status with no message.
  Status() = default;

  Status(iree_status_t status);

  // Creates a status with the specified code and error message.
  // If `code` is kOk, `message` is ignored.
  Status(StatusCode code, absl::string_view message);

  Status(const Status&);
  Status& operator=(const Status& x);

  ~Status();

  // Returns true if the Status is OK.
  IREE_MUST_USE_RESULT bool ok() const;

  // Returns the error code.
  StatusCode code() const;

  // Return a combination of the error code name and message.
  std::string ToString() const;

  // Ignores any errors, potentially suppressing complaints from any tools.
  void IgnoreError() {}

  friend bool operator==(const Status& lhs, const Status& rhs) {
    return lhs.code() == rhs.code();
  }
  friend bool operator!=(const Status& lhs, const Status& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator==(const Status& lhs, const StatusCode& rhs) {
    return lhs.code() == rhs;
  }
  friend bool operator!=(const Status& lhs, const StatusCode& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator==(const StatusCode& lhs, const Status& rhs) {
    return lhs == rhs.code();
  }
  friend bool operator!=(const StatusCode& lhs, const Status& rhs) {
    return !(lhs == rhs);
  }

 private:
  // TODO(#265): remove message().
  absl::string_view message() const;
  friend Status Annotate(const Status& s, absl::string_view msg);

  struct State {
    StatusCode code;
    std::string message;
  };
  // OK status has a nullptr state_.  Otherwise, 'state_' points to
  // a 'State' structure containing the error code and message(s).
  std::unique_ptr<State> state_;
};

// Returns an OK status, equivalent to a default constructed instance.
Status OkStatus();

// Prints a human-readable representation of `x` to `os`.
std::ostream& operator<<(std::ostream& os, const Status& x);

// Returns a Status that is identical to `s` except that the message()
// has been augmented by adding `msg` to the end of the original message.
Status Annotate(const Status& s, absl::string_view msg);

IREE_MUST_USE_RESULT static inline bool IsOk(const Status& status) {
  return status.code() == StatusCode::kOk;
}

IREE_MUST_USE_RESULT static inline bool IsOk(iree_status_t status) {
  return iree_status_is_ok(status);
}

IREE_MUST_USE_RESULT static inline bool IsAborted(const Status& status) {
  return status.code() == StatusCode::kAborted;
}

IREE_MUST_USE_RESULT static inline bool IsAlreadyExists(const Status& status) {
  return status.code() == StatusCode::kAlreadyExists;
}

IREE_MUST_USE_RESULT static inline bool IsCancelled(const Status& status) {
  return status.code() == StatusCode::kCancelled;
}

IREE_MUST_USE_RESULT static inline bool IsDataLoss(const Status& status) {
  return status.code() == StatusCode::kDataLoss;
}

IREE_MUST_USE_RESULT static inline bool IsDeadlineExceeded(
    const Status& status) {
  return status.code() == StatusCode::kDeadlineExceeded;
}

IREE_MUST_USE_RESULT static inline bool IsFailedPrecondition(
    const Status& status) {
  return status.code() == StatusCode::kFailedPrecondition;
}

IREE_MUST_USE_RESULT static inline bool IsInternal(const Status& status) {
  return status.code() == StatusCode::kInternal;
}

IREE_MUST_USE_RESULT static inline bool IsInvalidArgument(
    const Status& status) {
  return status.code() == StatusCode::kInvalidArgument;
}

IREE_MUST_USE_RESULT static inline bool IsNotFound(const Status& status) {
  return status.code() == StatusCode::kNotFound;
}

IREE_MUST_USE_RESULT static inline bool IsOutOfRange(const Status& status) {
  return status.code() == StatusCode::kOutOfRange;
}

IREE_MUST_USE_RESULT static inline bool IsPermissionDenied(
    const Status& status) {
  return status.code() == StatusCode::kPermissionDenied;
}

IREE_MUST_USE_RESULT static inline bool IsResourceExhausted(
    const Status& status) {
  return status.code() == StatusCode::kResourceExhausted;
}

IREE_MUST_USE_RESULT static inline bool IsUnauthenticated(
    const Status& status) {
  return status.code() == StatusCode::kUnauthenticated;
}

IREE_MUST_USE_RESULT static inline bool IsUnavailable(const Status& status) {
  return status.code() == StatusCode::kUnavailable;
}

IREE_MUST_USE_RESULT static inline bool IsUnimplemented(const Status& status) {
  return status.code() == StatusCode::kUnimplemented;
}

IREE_MUST_USE_RESULT static inline bool IsUnknown(const Status& status) {
  return status.code() == StatusCode::kUnknown;
}

// TODO(#265): rename to IREE_CHECK_OK and make compatible with C API macros.
#define CHECK_OK(val) CHECK_EQ(::iree::StatusCode::kOk, (val))
#define QCHECK_OK(val) QCHECK_EQ(::iree::StatusCode::kOk, (val))
#define DCHECK_OK(val) DCHECK_EQ(::iree::StatusCode::kOk, (val))

}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_H_
