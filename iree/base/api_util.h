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

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/time/time.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/base/shape.h"
#include "iree/base/status.h"

namespace iree {

inline iree_status_t ToApiStatus(const Status& status) {
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return iree_make_status(status.code());
}

inline Status FromApiStatus(iree_status_t status_code, SourceLocation loc) {
  return StatusBuilder(static_cast<StatusCode>(status_code), loc);
}

// Internal helper for concatenating macro values.
#define IREE_API_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define IREE_API_STATUS_MACROS_IMPL_CONCAT_(x, y) \
  IREE_API_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

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

// clang-format off
#define IREE_API_STATUS_MACROS_IMPL_ELSE_BLOCKER_ switch (0) case 0: default:  // NOLINT
// clang-format on

#define IREE_API_RETURN_IF_ERROR(expr)                         \
  IREE_API_STATUS_MACROS_IMPL_ELSE_BLOCKER_                    \
  if (::iree::status_macro_internal::StatusAdaptorForApiMacros \
          status_adaptor = {expr}) {                           \
  } else /* NOLINT */                                          \
    return ::iree::ToApiStatus(status_adaptor.Consume())

#define IREE_API_ASSIGN_OR_RETURN(...)                               \
  IREE_API_STATUS_MACROS_IMPL_GET_VARIADIC_(                         \
      (__VA_ARGS__, IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_, \
       IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_))             \
  (__VA_ARGS__)

#define IREE_API_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, \
                                                         ...)              \
  NAME
#define IREE_API_STATUS_MACROS_IMPL_GET_VARIADIC_(args) \
  IREE_API_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_ args

#define IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, std::move(_))
#define IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr,         \
                                                        error_expression)   \
  IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(                            \
      IREE_API_STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, \
      rexpr, error_expression)
#define IREE_API_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr, \
                                                      error_expression)     \
  auto statusor = (rexpr);                                                  \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                                 \
    return ::iree::ToApiStatus(std::move(statusor).status());               \
  }                                                                         \
  lhs = std::move(statusor).ValueOrDie()

// Converts an iree_time_t to its equivalent absl::Time.
inline absl::Time ToAbslTime(iree_time_t time) {
  if (time == IREE_TIME_INFINITE_PAST) {
    return absl::InfinitePast();
  } else if (time == IREE_TIME_INFINITE_FUTURE) {
    return absl::InfiniteFuture();
  } else {
    return absl::FromUnixNanos(time);
  }
}

// Converts a Shape to an iree_shape_t.
inline iree_status_t ToApiShape(const Shape& shape, iree_shape_t* out_shape) {
  out_shape->rank = shape.size();
  if (shape.size() > ABSL_ARRAYSIZE(out_shape->dims)) {
    return IREE_STATUS_OUT_OF_RANGE;
  }
  for (int i = 0; i < out_shape->rank; ++i) {
    out_shape->dims[i] = shape[i];
  }
  return IREE_STATUS_OK;
}

// Returns a vector initialized with the contents of a C-style list query.
// For functions of the form (..., capacity, out_values, out_count) this will
// try to fetch the items and resize as needed such that the returned value
// contains all items available.
//
// Returns the empty vector if the query fails for any reason.
template <typename T, typename... Args>
absl::InlinedVector<T, 4> QueryListValues(
    iree_status_t (*fn)(Args..., iree_host_size_t, T*, iree_host_size_t*),
    Args... args) {
  absl::InlinedVector<T, 4> values(4);
  iree_host_size_t count = 0;
  iree_status_t status = fn(args..., values.size(), values.data(), &count);
  if (iree_status_is_out_of_range(status)) {
    values.resize(count);
    status = fn(args..., values.size(), values.data(), &count);
  } else if (!iree_status_is_ok(status)) {
    return {};
  }
  values.resize(count);
  return values;
}

}  // namespace iree

#endif  // IREE_BASE_API_UTIL_H_
