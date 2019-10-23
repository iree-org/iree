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

#ifndef IREE_BASE_INTERNAL_STATUS_MACROS_H_
#define IREE_BASE_INTERNAL_STATUS_MACROS_H_

#include "base/internal/status.h"
#include "base/internal/status_builder.h"
#include "base/internal/statusor.h"
#include "base/source_location.h"

// Evaluates an expression that produces a `iree::Status`. If the status is not
// ok, returns it from the current function.
#define RETURN_IF_ERROR(expr)                                   \
  STATUS_MACROS_IMPL_ELSE_BLOCKER_                              \
  if (iree::status_macro_internal::StatusAdaptorForMacros       \
          status_macro_internal_adaptor = {(expr), IREE_LOC}) { \
  } else /* NOLINT */                                           \
    return status_macro_internal_adaptor.Consume()

// Executes an expression `rexpr` that returns a `iree::StatusOr<T>`. On OK,
// moves its value into the variable defined by `lhs`, otherwise returns
// from the current function.
#define ASSIGN_OR_RETURN(...)                                                \
  STATUS_MACROS_IMPL_GET_VARIADIC_((__VA_ARGS__,                             \
                                    STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_,  \
                                    STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_)) \
  (__VA_ARGS__)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

// MSVC incorrectly expands variadic macros, splice together a macro call to
// work around the bug.
#define STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define STATUS_MACROS_IMPL_GET_VARIADIC_(args) \
  STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_ args

#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, std::move(_))
#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, error_expression) \
  STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(                                      \
      STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, rexpr,    \
      error_expression)
#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr,      \
                                             error_expression)          \
  auto statusor = (rexpr);                                              \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                             \
    iree::StatusBuilder _(std::move(statusor).status(), IREE_LOC);      \
    (void)_; /* error_expression is allowed to not use this variable */ \
    return (error_expression);                                          \
  }                                                                     \
  lhs = std::move(statusor).ValueOrDie()

// Internal helper for concatenating macro values.
#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

// clang-format off
#define STATUS_MACROS_IMPL_ELSE_BLOCKER_ switch (0) case 0: default:  // NOLINT
// clang-format on

namespace iree {
namespace status_macro_internal {

// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
 public:
  StatusAdaptorForMacros(const Status& status, SourceLocation loc)
      : builder_(status, loc) {}

  StatusAdaptorForMacros(Status&& status, SourceLocation loc)
      : builder_(std::move(status), loc) {}

  StatusAdaptorForMacros(const StatusBuilder& builder, SourceLocation loc)
      : builder_(builder) {}

  StatusAdaptorForMacros(StatusBuilder&& builder, SourceLocation loc)
      : builder_(std::move(builder)) {}

  StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
  StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

  explicit operator bool() const { return ABSL_PREDICT_TRUE(builder_.ok()); }

  StatusBuilder&& Consume() { return std::move(builder_); }

 private:
  StatusBuilder builder_;
};

}  // namespace status_macro_internal
}  // namespace iree

#endif  // IREE_BASE_INTERNAL_STATUS_MACROS_H_
