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

#include <atomic>
#include <string>

#include "absl/base/attributes.h"
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "iree/base/internal/logging.h"

ABSL_DECLARE_FLAG(bool, iree_status_save_stack_trace);

namespace iree {

// True if Status objects will capture stack traces on init for non-ok Statuses.
bool DoesStatusSaveStackTrace();

// Enables/disables status stack trace saving. This is global for the process.
// While useful for debugging, stack traces can impact performance severely.
void StatusSavesStackTrace(bool on_off);

enum class StatusCode : int {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
  kDoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

std::string StatusCodeToString(StatusCode code);

class ABSL_MUST_USE_RESULT Status;

// A Status value can be either OK or not-OK
//   * OK indicates that the operation succeeded.
//   * A not-OK value indicates that the operation failed and contains details
//     about the error.
class Status final {
 public:
  // Creates an OK status with no message.
  Status();

  // Creates a status with the specified code and error message.
  Status(StatusCode code, absl::string_view message);

  Status(const Status&);
  Status& operator=(const Status& x);

  ~Status();

  // Returns true if the Status is OK.
  ABSL_MUST_USE_RESULT bool ok() const;

  // Returns the error code.
  StatusCode code() const;

  // Returns the error message. Note: prefer ToString() for debug logging.
  // This message rarely describes the error code. It is not unusual for the
  // error message to be the empty string.
  absl::string_view message() const;

  // Return a combination of the error code name and message.
  std::string ToString() const;

  // Compatibility with upstream API. Equiv to ToString().
  std::string error_message() const { return ToString(); }

  friend bool operator==(const Status&, const Status&);
  friend bool operator!=(const Status&, const Status&);

  // Ignores any errors, potentially suppressing complaints from any tools.
  void IgnoreError() const;

 private:
  static bool EqualsSlow(const Status& a, const Status& b);

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

#define CHECK_OK(val) CHECK_EQ(::iree::OkStatus(), (val))
#define QCHECK_OK(val) QCHECK_EQ(::iree::OkStatus(), (val))
#define DCHECK_OK(val) DCHECK_EQ(::iree::OkStatus(), (val))

}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_H_
