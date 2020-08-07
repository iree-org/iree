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

#include "iree/base/internal/status_errors.h"

namespace iree {

bool IsAborted(const Status& status) {
  return status.code() == StatusCode::kAborted;
}

bool IsAlreadyExists(const Status& status) {
  return status.code() == StatusCode::kAlreadyExists;
}

bool IsCancelled(const Status& status) {
  return status.code() == StatusCode::kCancelled;
}

bool IsDataLoss(const Status& status) {
  return status.code() == StatusCode::kDataLoss;
}

bool IsDeadlineExceeded(const Status& status) {
  return status.code() == StatusCode::kDeadlineExceeded;
}

bool IsFailedPrecondition(const Status& status) {
  return status.code() == StatusCode::kFailedPrecondition;
}

bool IsInternal(const Status& status) {
  return status.code() == StatusCode::kInternal;
}

bool IsInvalidArgument(const Status& status) {
  return status.code() == StatusCode::kInvalidArgument;
}

bool IsNotFound(const Status& status) {
  return status.code() == StatusCode::kNotFound;
}

bool IsOutOfRange(const Status& status) {
  return status.code() == StatusCode::kOutOfRange;
}

bool IsPermissionDenied(const Status& status) {
  return status.code() == StatusCode::kPermissionDenied;
}

bool IsResourceExhausted(const Status& status) {
  return status.code() == StatusCode::kResourceExhausted;
}

bool IsUnauthenticated(const Status& status) {
  return status.code() == StatusCode::kUnauthenticated;
}

bool IsUnavailable(const Status& status) {
  return status.code() == StatusCode::kUnavailable;
}

bool IsUnimplemented(const Status& status) {
  return status.code() == StatusCode::kUnimplemented;
}

bool IsUnknown(const Status& status) {
  return status.code() == StatusCode::kUnknown;
}

}  // namespace iree
