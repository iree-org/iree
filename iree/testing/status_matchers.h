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

#ifndef IREE_TESTING_STATUS_MATCHERS_H_
#define IREE_TESTING_STATUS_MATCHERS_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "iree/base/status.h"
#include "iree/testing/gtest.h"

namespace iree {

namespace internal {

// Implements a gMock matcher that checks that an iree::StaturOr<T> has an OK
// status and that the contained T value matches another matcher.
template <typename T>
class IsOkAndHoldsMatcher
    : public ::testing::MatcherInterface<const StatusOr<T> &> {
 public:
  template <typename MatcherT>
  IsOkAndHoldsMatcher(MatcherT &&value_matcher)
      : value_matcher_(::testing::SafeMatcherCast<const T &>(value_matcher)) {}

  // From testing::MatcherInterface.
  void DescribeTo(std::ostream *os) const override {
    *os << "is OK and contains a value that ";
    value_matcher_.DescribeTo(os);
  }

  // From testing::MatcherInterface.
  void DescribeNegationTo(std::ostream *os) const override {
    *os << "is not OK or contains a value that ";
    value_matcher_.DescribeNegationTo(os);
  }

  // From testing::MatcherInterface.
  bool MatchAndExplain(
      const StatusOr<T> &status_or,
      ::testing::MatchResultListener *listener) const override {
    if (!status_or.ok()) {
      *listener << "which is not OK";
      return false;
    }

    ::testing::StringMatchResultListener value_listener;
    bool is_a_match =
        value_matcher_.MatchAndExplain(status_or.value(), &value_listener);
    std::string value_explanation = value_listener.str();
    if (!value_explanation.empty()) {
      *listener << "which contains a value " << value_explanation;
    }

    return is_a_match;
  }

 private:
  const ::testing::Matcher<const T &> value_matcher_;
};

// A polymorphic IsOkAndHolds() matcher.
//
// IsOkAndHolds() returns a matcher that can be used to process an IsOkAndHolds
// expectation. However, the value type T is not provided when IsOkAndHolds() is
// invoked. The value type is only inferable when the gUnit framework invokes
// the matcher with a value. Consequently, the IsOkAndHolds() function must
// return an object that is implicitly convertible to a matcher for StatusOr<T>.
// gUnit refers to such an object as a polymorphic matcher, since it can be used
// to match with more than one type of value.
template <typename ValueMatcherT>
class IsOkAndHoldsGenerator {
 public:
  explicit IsOkAndHoldsGenerator(ValueMatcherT value_matcher)
      : value_matcher_(std::move(value_matcher)) {}

  template <typename T>
  operator ::testing::Matcher<const StatusOr<T> &>() const {
    return ::testing::MakeMatcher(new IsOkAndHoldsMatcher<T>(value_matcher_));
  }

 private:
  const ValueMatcherT value_matcher_;
};

// Implements a gMock matcher for checking error-code expectations on
// iree::Status and iree::StatusOr objects.
template <typename Enum, typename Matchee>
class StatusMatcher : public ::testing::MatcherInterface<Matchee> {
 public:
  StatusMatcher(Enum code, absl::optional<absl::string_view> message)
      : code_(code), message_(message) {}

  // From testing::MatcherInterface.
  //
  // Describes the expected error code.
  void DescribeTo(std::ostream *os) const override {
    *os << "error code " << StatusCodeToString(code_);
    if (message_.has_value()) {
      *os << "::'" << message_.value() << "'";
    }
  }

  // From testing::MatcherInterface.
  //
  // Tests whether |matchee| has an error code that meets this matcher's
  // expectation. If an error message string is specified in this matcher, it
  // also tests that |matchee| has an error message that matches that
  // expectation.
  bool MatchAndExplain(
      Matchee &matchee,
      ::testing::MatchResultListener *listener) const override {
    if (GetCode(matchee) != code_) {
      *listener << "whose error code is "
                << StatusCodeToString(GetCode(matchee));
      return false;
    }
    if (message_.has_value() && GetMessage(matchee) != message_.value()) {
      *listener << "whose error message is '" << GetMessage(matchee) << "'";
      return false;
    }
    return true;
  }

 private:
  template <typename T>
  StatusCode GetCode(const T &matchee) const {
    return GetCode(matchee.status());
  }

  StatusCode GetCode(const iree_status_code_t &status_code) const {
    return static_cast<StatusCode>(status_code);
  }

  StatusCode GetCode(const iree_status_t &status) const {
    return static_cast<StatusCode>(iree_status_code(status));
  }

  StatusCode GetCode(const Status &status) const { return status.code(); }

  template <typename T>
  std::string GetMessage(const T &matchee) const {
    return GetMessage(matchee.status());
  }

  std::string GetMessage(const iree_status_t &status) const {
    return Status::ToString(status);
  }

  std::string GetMessage(const Status &status) const {
    return status.ToString();
  }

  // Expected error code.
  const Enum code_;

  // Expected error message (empty if none expected and verified).
  const absl::optional<std::string> message_;
};

// StatusMatcherGenerator is an intermediate object returned by
// iree::testing::status::StatusIs().
// It implements implicit type-cast operators to supported matcher types:
// Matcher<const Status &> and Matcher<const StatusOr<T> &>. These typecast
// operators create gMock matchers that test OK expectations on a status
// container.
template <typename Enum>
class StatusIsMatcherGenerator {
 public:
  StatusIsMatcherGenerator(Enum code, absl::optional<absl::string_view> message)
      : code_(code), message_(message) {}

  operator ::testing::Matcher<const StatusCode &>() const {
    return ::testing::MakeMatcher(
        new internal::StatusMatcher<Enum, const StatusCode &>(code_, message_));
  }

  operator ::testing::Matcher<const iree_status_t &>() const {
    return ::testing::MakeMatcher(
        new internal::StatusMatcher<Enum, const iree_status_t &>(code_,
                                                                 message_));
  }

  operator ::testing::Matcher<const Status &>() const {
    return ::testing::MakeMatcher(
        new internal::StatusMatcher<Enum, const Status &>(code_, message_));
  }

  template <class T>
  operator ::testing::Matcher<const StatusOr<T> &>() const {
    return ::testing::MakeMatcher(
        new internal::StatusMatcher<Enum, const StatusOr<T> &>(code_,
                                                               message_));
  }

 private:
  // Expected error code.
  const Enum code_;

  // Expected error message (empty if none expected and verified).
  const absl::optional<std::string> message_;
};

// Implements a gMock matcher that checks whether a status container (e.g.
// iree::Status or iree::StatusOr<T>) has an OK status.
template <class T>
class IsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  IsOkMatcherImpl() = default;

  // From testing::MatcherInterface.
  //
  // Describes the OK expectation.
  void DescribeTo(std::ostream *os) const override { *os << "is OK"; }

  // From testing::MatcherInterface.
  //
  // Describes the negative OK expectation.
  void DescribeNegationTo(std::ostream *os) const override {
    *os << "is not OK";
  }

  // From testing::MatcherInterface.
  //
  // Tests whether |status_container|'s OK value meets this matcher's
  // expectation.
  bool MatchAndExplain(
      const T &status_container,
      ::testing::MatchResultListener *listener) const override {
    if (!::iree::IsOk(status_container)) {
      *listener << "which is not OK";
      return false;
    }
    return true;
  }
};

// IsOkMatcherGenerator is an intermediate object returned by iree::IsOk().
// It implements implicit type-cast operators to supported matcher types:
// Matcher<const Status &> and Matcher<const StatusOr<T> &>. These typecast
// operators create gMock matchers that test OK expectations on a status
// container.
class IsOkMatcherGenerator {
 public:
  operator ::testing::Matcher<const iree_status_t &>() const {
    return ::testing::MakeMatcher(
        new internal::IsOkMatcherImpl<const iree_status_t &>());
  }

  operator ::testing::Matcher<const Status &>() const {
    return ::testing::MakeMatcher(
        new internal::IsOkMatcherImpl<const Status &>());
  }

  template <class T>
  operator ::testing::Matcher<const StatusOr<T> &>() const {
    return ::testing::MakeMatcher(
        new internal::IsOkMatcherImpl<const StatusOr<T> &>());
  }
};

}  // namespace internal

namespace testing {
namespace status {

// Returns a gMock matcher that expects an iree::StatusOr<T> object to have an
// OK status and for the contained T object to match |value_matcher|.
//
// Example:
//
//     StatusOr<string> raven_speech_result = raven.Speak();
//     EXPECT_THAT(raven_speech_result, IsOkAndHolds(HasSubstr("nevermore")));
//
// If foo is an object of type T and foo_result is an object of type
// StatusOr<T>, you can write:
//
//     EXPECT_THAT(foo_result, IsOkAndHolds(foo));
//
// instead of:
//
//     EXPECT_THAT(foo_result, IsOkAndHolds(Eq(foo)));
template <typename ValueMatcherT>
internal::IsOkAndHoldsGenerator<ValueMatcherT> IsOkAndHolds(
    ValueMatcherT value_matcher) {
  return internal::IsOkAndHoldsGenerator<ValueMatcherT>(value_matcher);
}

// Returns a gMock matcher that expects an iree::Status object to have the
// given |code|.
template <typename Enum>
internal::StatusIsMatcherGenerator<Enum> StatusIs(Enum code) {
  return internal::StatusIsMatcherGenerator<Enum>(code, absl::nullopt);
}

// Returns a gMock matcher that expects an iree::Status object to have the
// given |code| and |message|.
template <typename Enum>
internal::StatusIsMatcherGenerator<Enum> StatusIs(Enum code,
                                                  absl::string_view message) {
  return internal::StatusIsMatcherGenerator<Enum>(code, message);
}

// Returns an internal::IsOkMatcherGenerator, which may be typecast to a
// Matcher<iree::Status> or Matcher<iree::StatusOr<T>>. These gMock
// matchers test that a given status container has an OK status.
inline internal::IsOkMatcherGenerator IsOk() {
  return internal::IsOkMatcherGenerator();
}

}  // namespace status
}  // namespace testing

// Macros for testing the results of functions that return iree::Status or
// iree::StatusOr<T> (for any type T).
#define IREE_EXPECT_OK(rexpr) \
  EXPECT_THAT(rexpr, ::iree::testing::status::IsOk())
#define IREE_ASSERT_OK(rexpr) \
  ASSERT_THAT(rexpr, ::iree::testing::status::IsOk())
#define IREE_EXPECT_STATUS_IS(expected_code, expr)     \
  EXPECT_THAT(expr, ::iree::testing::status::StatusIs( \
                        static_cast<::iree::StatusCode>(expected_code)))

// Executes an expression that returns an iree::StatusOr<T>, and assigns the
// contained variable to lhs if the error code is OK.
// If the Status is non-OK, generates a test failure and returns from the
// current function, which must have a void return type.
//
// Example: Assigning to an existing value
//   IREE_ASSERT_OK_AND_ASSIGN(ValueType value, MaybeGetValue(arg));
//
// The value assignment example might expand into:
//   StatusOr<ValueType> status_or_value = MaybeGetValue(arg);
//   IREE_ASSERT_OK(status_or_value.status());
//   ValueType value = status_or_value.value();
#define IREE_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                             \
  IREE_ASSERT_OK_AND_ASSIGN_IMPL(                                         \
      IREE_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr);

#define IREE_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                   \
  IREE_ASSERT_OK(statusor.status());                         \
  lhs = std::move(statusor.value())
#define IREE_STATUS_MACROS_CONCAT_NAME(x, y) \
  IREE_STATUS_MACROS_CONCAT_IMPL(x, y)
#define IREE_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

// Implements the PrintTo() method for iree::StatusOr<T>. This method is
// used by gUnit to print iree::StatusOr<T> objects for debugging. The
// implementation relies on gUnit for printing values of T when a
// iree::StatusOr<T> object is OK and contains a value.
template <typename T>
void PrintTo(const StatusOr<T> &statusor, std::ostream *os) {
  if (!statusor.ok()) {
    *os << statusor.status();
  } else {
    *os << "OK: " << ::testing::PrintToString(statusor.value());
  }
}

}  // namespace iree

#endif  // IREE_TESTING_STATUS_MATCHERS_H_
