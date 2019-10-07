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

#ifndef IREE_BASE_API_UTIL_H_
#define IREE_BASE_API_UTIL_H_

#include "iree/base/api.h"
#include "iree/base/status.h"

namespace iree {

constexpr iree_status_t ToApiStatus(StatusCode status_code) {
  return static_cast<iree_status_t>(status_code);
}

// clang-format off
#define IREE_API_STATUS_MACROS_IMPL_ELSE_BLOCKER_ switch (0) case 0: default:  // NOLINT
// clang-format on

namespace status_macro_internal {
class StatusAdaptorForApiMacros {
 public:
  StatusAdaptorForApiMacros(const Status& status) : status_(status) {}
  StatusAdaptorForApiMacros(Status&& status) : status_(std::move(status)) {}
  StatusAdaptorForApiMacros(const StatusAdaptorForApiMacros&) = delete;
  StatusAdaptorForApiMacros& operator=(const StatusAdaptorForApiMacros&) =
      delete;
  explicit operator bool() const { return ABSL_PREDICT_TRUE(status_.ok()); }
  Status&& Consume() { return std::move(status_); }

 private:
  Status status_;
};
}  // namespace status_macro_internal

#define IREE_API_RETURN_IF_ERROR(expr)                         \
  IREE_API_STATUS_MACROS_IMPL_ELSE_BLOCKER_                    \
  if (::iree::status_macro_internal::StatusAdaptorForApiMacros \
          status_adaptor = (expr)) {                           \
  } else /* NOLINT */                                          \
    return ::iree::ToApiStatus(status_adaptor.Consume().code())

#define IREE_API_RETURN_IF_API_ERROR(expr)  \
  IREE_API_STATUS_MACROS_IMPL_ELSE_BLOCKER_ \
  if (iree_status_t status = (expr)) {      \
  } else /* NOLINT */                       \
    return status

}  // namespace iree

#endif  // IREE_BASE_API_UTIL_H_
