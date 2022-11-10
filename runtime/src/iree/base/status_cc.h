// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_STATUS_CC_H_
#define IREE_BASE_STATUS_CC_H_

#ifndef __cplusplus
#error iree::Status is only usable in C++ code.
#endif  // !__cplusplus

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/attributes.h"
#include "iree/base/target_platform.h"

namespace iree {

namespace status_impl {

template <class T, class U = T>
constexpr T exchange(T& obj, U&& new_value) {
  T old_value = std::move(obj);
  obj = std::forward<U>(new_value);
  return old_value;
}

}  // namespace status_impl

//===----------------------------------------------------------------------===//
// Status codes and source location utilities
//===----------------------------------------------------------------------===//

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
  kDeferred = IREE_STATUS_DEFERRED,
};

static inline const char* StatusCodeToString(StatusCode code) {
  return iree_status_code_string(static_cast<iree_status_code_t>(code));
}

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

class IREE_MUST_USE_RESULT Status;

// A Status value can be either OK or not-OK
//   * OK indicates that the operation succeeded.
//   * A not-OK value indicates that the operation failed and contains
//   status_impls
//     about the error.
class Status final {
 public:
  // Return a combination of the error code name and message.
  IREE_MUST_USE_RESULT static inline std::string ToString(
      iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return "OK";
    }
    iree_host_size_t buffer_length = 0;
    if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                          /*buffer=*/NULL, &buffer_length))) {
      return "<!>";
    }
    std::string result(buffer_length, '\0');
    if (IREE_UNLIKELY(!iree_status_format(status, result.size() + 1,
                                          const_cast<char*>(result.data()),
                                          &buffer_length))) {
      return "<!>";
    }
    return result;
  }

  // Creates an OK status with no message.
  Status() = default;

  // Takes ownership of a C API status instance.
  Status(iree_status_t&& status) noexcept
      : value_(status_impl::exchange(
            status, iree_status_from_code(iree_status_code(status)))) {}

  // Takes ownership of a C API status instance wrapped in a Status.
  Status(Status& other) noexcept
      : value_(status_impl::exchange(other.value_,
                                     iree_status_from_code(other.code()))) {}
  Status(Status&& other) noexcept
      : value_(status_impl::exchange(other.value_,
                                     iree_status_from_code(other.code()))) {}
  Status& operator=(Status&& other) {
    if (this != &other) {
      if (IREE_UNLIKELY(value_)) iree_status_ignore(value_);
      value_ = status_impl::exchange(other.value_,
                                     iree_status_from_code(other.code()));
    }
    return *this;
  }

  Status(iree_status_code_t code) : value_(iree_status_from_code(code)) {}
  Status& operator=(const iree_status_code_t& code) {
    if (IREE_UNLIKELY(value_)) iree_status_ignore(value_);
    value_ = iree_status_from_code(code);
    return *this;
  }

  Status(StatusCode code) : value_(iree_status_from_code(code)) {}
  Status& operator=(const StatusCode& code) {
    if (IREE_UNLIKELY(value_)) iree_status_ignore(value_);
    value_ = iree_status_from_code(code);
    return *this;
  }

  // Creates a status with the specified code and error message.
  // If `code` is kOk, `message` is ignored.
  Status(StatusCode code, const char* message) {
    if (IREE_UNLIKELY(code != StatusCode::kOk)) {
      value_ = (!message || !strlen(message))
                   ? iree_status_from_code(code)
                   : iree_status_allocate(static_cast<iree_status_code_t>(code),
                                          /*file=*/nullptr, /*line=*/0,
                                          iree_make_cstring_view(message));
    }
  }
  Status(StatusCode code, SourceLocation location, const char* message) {
    if (IREE_UNLIKELY(code != StatusCode::kOk)) {
      value_ = iree_status_allocate(static_cast<iree_status_code_t>(code),
                                    location.file_name(), location.line(),
                                    iree_make_cstring_view(message));
    }
  }

  ~Status() {
    if (IREE_UNLIKELY((uintptr_t)(value_) & ~IREE_STATUS_CODE_MASK)) {
      iree_status_free(value_);
    }
  }

  // Returns true if the Status is OK.
  IREE_MUST_USE_RESULT bool ok() const { return iree_status_is_ok(value_); }

  // Returns the error code.
  IREE_MUST_USE_RESULT StatusCode code() const {
    return static_cast<StatusCode>(iree_status_code(value_));
  }

  // Return a combination of the error code name and message.
  IREE_MUST_USE_RESULT std::string ToString() const {
    return Status::ToString(value_);
  }

  // Ignores any errors, potentially suppressing complaints from any tools.
  void IgnoreError() { value_ = iree_status_ignore(value_); }

  // Converts to a C API status instance and transfers ownership.
  IREE_MUST_USE_RESULT operator iree_status_t() && {
    return status_impl::exchange(
        value_, iree_status_from_code(iree_status_code(value_)));
  }

  iree_status_t get() const { return value_; }

  IREE_MUST_USE_RESULT iree_status_t release() {
    return status_impl::exchange(value_, iree_ok_status());
  }

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
  iree_status_t value_ = iree_ok_status();
};

// Returns an OK status, equivalent to a default constructed instance.
IREE_MUST_USE_RESULT static inline Status OkStatus() { return Status(); }

IREE_MUST_USE_RESULT static inline bool IsOk(const Status& status) {
  return status.code() == StatusCode::kOk;
}

IREE_MUST_USE_RESULT static inline bool IsOk(const iree_status_t& status) {
  return iree_status_is_ok(status);
}

//===----------------------------------------------------------------------===//
// StatusOr<T>
//===----------------------------------------------------------------------===//

template <typename T>
class IREE_MUST_USE_RESULT StatusOr;

namespace status_impl {

// https://en.cppreference.com/w/cpp/types/conjunction
template <typename... Ts>
struct conjunction : std::true_type {};
template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};
template <typename T>
struct conjunction<T> : T {};

// https://en.cppreference.com/w/cpp/types/disjunction
template <typename... Ts>
struct disjunction : std::false_type {};
template <typename T, typename... Ts>
struct disjunction<T, Ts...>
    : std::conditional<T::value, T, disjunction<Ts...>>::type {};
template <typename T>
struct disjunction<T> : T {};

// https://en.cppreference.com/w/cpp/utility/in_place
struct in_place_t {
  explicit in_place_t() = default;
};
/*inline*/ constexpr in_place_t in_place{};

// https://en.cppreference.com/w/cpp/types/negation
template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

template <typename T, typename U>
using IsStatusOrConversionAmbiguous =
    status_impl::disjunction<std::is_constructible<T, StatusOr<U>&>,
                             std::is_constructible<T, const StatusOr<U>&>,
                             std::is_constructible<T, StatusOr<U>&&>,
                             std::is_constructible<T, const StatusOr<U>&&>,
                             std::is_convertible<StatusOr<U>&, T>,
                             std::is_convertible<const StatusOr<U>&, T>,
                             std::is_convertible<StatusOr<U>&&, T>,
                             std::is_convertible<const StatusOr<U>&&, T>>;

template <typename T, typename U>
using IsStatusOrConversionAssigmentAmbiguous =
    status_impl::disjunction<IsStatusOrConversionAmbiguous<T, U>,
                             std::is_assignable<T&, StatusOr<U>&>,
                             std::is_assignable<T&, const StatusOr<U>&>,
                             std::is_assignable<T&, StatusOr<U>&&>,
                             std::is_assignable<T&, const StatusOr<U>&&>>;

template <typename T, typename U>
struct IsAmbiguousStatusOrForInitialization
    :  // Strip const-value refs from type and check again, else false_type.
       public std::conditional_t<
           std::is_same<std::remove_cv_t<std::remove_reference_t<U>>, U>::value,
           std::false_type,
           IsAmbiguousStatusOrForInitialization<
               T, std::remove_cv_t<std::remove_reference_t<U>>>> {};

template <typename T, typename U>
struct IsAmbiguousStatusOrForInitialization<T, StatusOr<U>>
    : public IsStatusOrConversionAmbiguous<T, U> {};

template <typename T, typename U>
using IsStatusOrDirectInitializationAmbiguous = status_impl::disjunction<
    std::is_same<StatusOr<T>, std::remove_cv_t<std::remove_reference_t<U>>>,
    std::is_same<Status, std::remove_cv_t<std::remove_reference_t<U>>>,
    std::is_same<status_impl::in_place_t,
                 std::remove_cv_t<std::remove_reference_t<U>>>,
    IsAmbiguousStatusOrForInitialization<T, U>>;

template <typename T, typename U>
using IsStatusOrDirectInitializationValid = status_impl::disjunction<
    // The is_same allows nested status ors to ignore this check iff same type.
    std::is_same<T, std::remove_cv_t<std::remove_reference_t<U>>>,
    status_impl::negation<IsStatusOrDirectInitializationAmbiguous<T, U>>>;

// Construct an instance of T in `p` through placement new, passing Args... to
// the constructor.
// This abstraction is here mostly for the gcc performance fix.
template <typename T, typename... Args>
void PlacementNew(void* p, Args&&... args) {
#if defined(__GNUC__) && !defined(__clang__)
  // Teach gcc that 'p' cannot be null, fixing code size issues.
  if (p == nullptr) __builtin_unreachable();
#endif
  new (p) T(std::forward<Args>(args)...);
}

// Helper base class to hold the data and all operations.
// We move all this to a base class to allow mixing with the appropriate
// TraitsBase specialization.
template <typename T>
class StatusOrData {
  template <typename U>
  friend class StatusOrData;

 public:
  StatusOrData() = delete;

  StatusOrData(const StatusOrData& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  StatusOrData(StatusOrData&& other) noexcept {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(status_impl::exchange(other.status_, other.status_.code()));
    }
  }

  template <typename U>
  explicit StatusOrData(const StatusOrData<U>& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  template <typename U>
  explicit StatusOrData(StatusOrData<U>&& other) {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(status_impl::exchange(other.status_, other.status_.code()));
    }
  }

  template <typename... Args>
  explicit StatusOrData(status_impl::in_place_t, Args&&... args)
      : data_(std::forward<Args>(args)...) {
    MakeStatus();
  }

  explicit StatusOrData(const T& value) : data_(value) { MakeStatus(); }
  explicit StatusOrData(T&& value) : data_(std::move(value)) { MakeStatus(); }

  explicit StatusOrData(Status&& status)
      : status_(status_impl::exchange(status, status.code())) {
    EnsureNotOk();
  }

  StatusOrData& operator=(const StatusOrData& other) {
    if (this == &other) return *this;
    if (other.ok()) {
      Assign(other.data_);
    } else {
      Assign(other.status_);
    }
    return *this;
  }

  StatusOrData& operator=(StatusOrData&& other) {
    if (this == &other) return *this;
    if (other.ok()) {
      Assign(std::move(other.data_));
    } else {
      Assign(status_impl::exchange(other.status_, other.status_.code()));
    }
    return *this;
  }

  ~StatusOrData() {
    if (ok()) {
      status_.~Status();
      data_.~T();
    } else {
      status_.~Status();
    }
  }

  void Assign(const T& value) {
    if (ok()) {
      data_.~T();
      MakeValue(value);
    } else {
      MakeValue(value);
      status_ = StatusCode::kOk;
    }
  }

  void Assign(T&& value) {
    if (ok()) {
      data_.~T();
      MakeValue(std::move(value));
    } else {
      MakeValue(std::move(value));
      status_ = StatusCode::kOk;
    }
  }

  void Assign(Status&& status) {
    Clear();
    status_ = status_impl::exchange(status, status.code());
    EnsureNotOk();
  }

  bool ok() const { return status_.ok(); }

 protected:
  // status_ will always be active after the constructor.
  // Union to be able to initialize exactly how we need without waste.
  // Eg. in the copy constructor we use the default constructor of Status in
  // the ok() path to avoid an extra Ref call.
  union {
    Status status_;
  };

  // data_ is active iff status_.ok()==true
  struct Dummy {};
  union {
    // When T is const, we need some non-const object we can cast to void* for
    // the placement new. dummy_ is that object.
    Dummy dummy_;
    T data_;
  };

  void Clear() {
    if (ok()) data_.~T();
  }

  void EnsureOk() const {
    if (IREE_UNLIKELY(!ok())) {
      iree_status_abort(status_.get());
    }
  }

  void EnsureNotOk() {
    if (IREE_UNLIKELY(ok())) {
      iree_status_abort(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "an OK status is not a valid constructor argument to StatusOr<T>"));
    }
  }

  // Construct the value (data_) through placement new with the passed arg.
  template <typename Arg>
  void MakeValue(Arg&& arg) {
    status_impl::PlacementNew<T>(&dummy_, std::forward<Arg>(arg));
  }

  // Construct the status (status_) through placement new with the passed arg.
  template <typename... Args>
  void MakeStatus(Args&&... args) {
    status_impl::PlacementNew<Status>(&status_, std::forward<Args>(args)...);
  }
};

// Helper base class to allow implicitly deleted constructors and assignment
// operations in StatusOr.
// TraitsBase will explicitly delete what it can't support and StatusOr will
// inherit that behavior implicitly.
template <bool Copy, bool Move>
struct TraitsBase {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = default;
  TraitsBase(TraitsBase&&) = default;
  TraitsBase& operator=(const TraitsBase&) = default;
  TraitsBase& operator=(TraitsBase&&) = default;
};

template <>
struct TraitsBase<false, true> {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = delete;
  TraitsBase(TraitsBase&&) = default;
  TraitsBase& operator=(const TraitsBase&) = delete;
  TraitsBase& operator=(TraitsBase&&) = default;
};

template <>
struct TraitsBase<false, false> {
  TraitsBase() = default;
  TraitsBase(const TraitsBase&) = delete;
  TraitsBase(TraitsBase&&) = delete;
  TraitsBase& operator=(const TraitsBase&) = delete;
  TraitsBase& operator=(TraitsBase&&) = delete;
};

}  // namespace status_impl

// StatusOr<T> is the union of a Status object and a T object.
//
// A StatusOr object either holds a usable value, or an error Status explaining
// why such a value is not present.
template <typename T>
class StatusOr
    : private status_impl::StatusOrData<T>,
      private status_impl::TraitsBase<std::is_copy_constructible<T>::value,
                                      std::is_move_constructible<T>::value> {
  template <typename U>
  friend class StatusOr;

  typedef status_impl::StatusOrData<T> Base;

 public:
  typedef T element_type;

  // Constructs a new StatusOr with StatusCode::kUnknown status.
  explicit StatusOr();

  // StatusOr<T> is copy constructible/assignable if T is copy constructible.
  StatusOr(const StatusOr&) = default;
  StatusOr& operator=(const StatusOr&) = default;

  // StatusOr<T> is move constructible/assignable if T is move constructible.
  StatusOr(StatusOr&&) = default;
  StatusOr& operator=(StatusOr&&) = default;

  // Converting constructors from StatusOr<U>, when T is constructible from U.
  // To avoid ambiguity, they are disabled if T is also constructible from
  // StatusOr<U>. Explicit iff the corresponding construction of T from U is
  // explicit.
  template <
      typename U,
      std::enable_if_t<
          status_impl::conjunction<
              status_impl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U&>,
              std::is_convertible<const U&, T>,
              status_impl::negation<
                  status_impl::IsStatusOrConversionAmbiguous<T, U>>>::value,
          int> = 0>
  StatusOr(const StatusOr<U>& other)  // NOLINT
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}
  template <
      typename U,
      std::enable_if_t<
          status_impl::conjunction<
              status_impl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U&>,
              status_impl::negation<std::is_convertible<const U&, T>>,
              status_impl::negation<
                  status_impl::IsStatusOrConversionAmbiguous<T, U>>>::value,
          int> = 0>
  explicit StatusOr(const StatusOr<U>& other)
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}

  template <
      typename U,
      std::enable_if_t<
          status_impl::conjunction<
              status_impl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U&&>, std::is_convertible<U&&, T>,
              status_impl::negation<
                  status_impl::IsStatusOrConversionAmbiguous<T, U>>>::value,
          int> = 0>
  StatusOr(StatusOr<U>&& other)  // NOLINT
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}
  template <
      typename U,
      std::enable_if_t<
          status_impl::conjunction<
              status_impl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U&&>,
              status_impl::negation<std::is_convertible<U&&, T>>,
              status_impl::negation<
                  status_impl::IsStatusOrConversionAmbiguous<T, U>>>::value,
          int> = 0>
  explicit StatusOr(StatusOr<U>&& other)
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}

  // Conversion copy/move assignment operator, T must be constructible and
  // assignable from U. Only enable if T cannot be directly assigned from
  // StatusOr<U>.
  template <typename U,
            std::enable_if_t<
                status_impl::conjunction<
                    status_impl::negation<std::is_same<T, U>>,
                    std::is_constructible<T, const U&>,
                    std::is_assignable<T, const U&>,
                    status_impl::negation<
                        status_impl::IsStatusOrConversionAssigmentAmbiguous<
                            T, U>>>::value,
                int> = 0>
  StatusOr& operator=(const StatusOr<U>& other) {
    this->Assign(other);
    return *this;
  }
  template <typename U,
            std::enable_if_t<
                status_impl::conjunction<
                    status_impl::negation<std::is_same<T, U>>,
                    std::is_constructible<T, U&&>, std::is_assignable<T, U&&>,
                    status_impl::negation<
                        status_impl::IsStatusOrConversionAssigmentAmbiguous<
                            T, U>>>::value,
                int> = 0>
  StatusOr& operator=(StatusOr<U>&& other) {
    this->Assign(std::move(other));
    return *this;
  }

  // Constructs a new StatusOr with the given value. After calling this
  // constructor, this->ok() will be true and the contained value may be
  // retrieved with value(), operator*(), or operator->().
  StatusOr(const T& value);

  // Takes ownership of a C API status instance.
  StatusOr(iree_status_t&& status) noexcept
      : Base(status_impl::exchange(
            status, iree_status_from_code(iree_status_code(status)))) {}

  // Constructs a new StatusOr with the given non-ok status. After calling this
  // constructor, this->ok() will be false and calls to value() will
  // IREE_CHECK-fail.
  StatusOr(const Status& status);
  StatusOr& operator=(const Status& status);

  // Similar to the `const T&` overload.
  //
  // REQUIRES: T is move constructible.
  StatusOr(T&& value);

  // RValue versions of the operations declared above.
  StatusOr(Status&& status);
  StatusOr& operator=(Status&& status);

  // Constructs the inner value T in-place using the provided args, using the
  // T(args...) constructor.
  template <typename... Args>
  explicit StatusOr(status_impl::in_place_t, Args&&... args);
  template <typename U, typename... Args>
  explicit StatusOr(status_impl::in_place_t, std::initializer_list<U> ilist,
                    Args&&... args);

  // Constructs the inner value T in-place using the provided args, using the
  // T(U) (direct-initialization) constructor. Only valid if T can be
  // constructed from a U. Can accept move or copy constructors. Explicit it
  // U is not convertible to T. To avoid ambiguity, this is disabled if U is
  // a StatusOr<J>, where J is convertible to T.
  template <typename U = T,
            std::enable_if_t<
                status_impl::conjunction<
                    status_impl::IsStatusOrDirectInitializationValid<T, U&&>,
                    std::is_constructible<T, U&&>,
                    std::is_convertible<U&&, T>>::value,
                int> = 0>
  StatusOr(U&& u)  // NOLINT
      : StatusOr(status_impl::in_place, std::forward<U>(u)) {}

  template <typename U = T,
            std::enable_if_t<
                status_impl::conjunction<
                    status_impl::IsStatusOrDirectInitializationValid<T, U&&>,
                    std::is_constructible<T, U&&>,
                    status_impl::negation<std::is_convertible<U&&, T>>>::value,
                int> = 0>
  explicit StatusOr(U&& u)  // NOLINT
      : StatusOr(status_impl::in_place, std::forward<U>(u)) {}

  // Returns this->ok()
  explicit operator bool() const { return ok(); }

  // Returns this->status().ok()
  IREE_MUST_USE_RESULT bool ok() const { return this->status_.ok(); }

  // Returns a reference to our status. If this contains a T, then
  // returns OkStatus().
  const Status& status() const&;
  Status status() &&;

  // Returns a reference to the held value if `this->ok()`, or IREE_CHECK-fails.
  // If you have already checked the status using `this->ok()` or
  // `operator bool()`, you probably want to use `operator*()` or `operator->()`
  // to access the value instead of `value`.
  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  // Returns a reference to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  // Returns a pointer to the current value.
  //
  // REQUIRES: this->ok() == true, otherwise the behavior is undefined.
  const T* operator->() const;
  T* operator->();

  // Returns a copy of the current value if this->ok() == true. Otherwise
  // returns a default value.
  template <typename U>
  T value_or(U&& default_value) const&;
  template <typename U>
  T value_or(U&& default_value) &&;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  using status_impl::StatusOrData<T>::Assign;
  template <typename U>
  void Assign(const StatusOr<U>& other);
  template <typename U>
  void Assign(StatusOr<U>&& other);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation status_impls for StatusOr<T>

template <typename T>
StatusOr<T>::StatusOr() : Base(Status(StatusCode::kUnknown, "")) {}

template <typename T>
StatusOr<T>::StatusOr(const T& value) : Base(value) {}

template <typename T>
StatusOr<T>::StatusOr(T&& value) : Base(std::move(value)) {}

template <typename T>
StatusOr<T>::StatusOr(const Status& status) : Base(status) {}

template <typename T>
StatusOr<T>::StatusOr(Status&& status) : Base(std::move(status)) {}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(const Status& status) {
  this->Assign(status);
  return *this;
}

template <typename T>
StatusOr<T>& StatusOr<T>::operator=(Status&& status) {
  this->Assign(std::move(status));
  return *this;
}

template <typename T>
template <typename U>
inline void StatusOr<T>::Assign(const StatusOr<U>& other) {
  if (other.ok()) {
    this->Assign(other.value());
  } else {
    this->Assign(other.status());
  }
}

template <typename T>
template <typename U>
inline void StatusOr<T>::Assign(StatusOr<U>&& other) {
  if (other.ok()) {
    this->Assign(std::move(other).value());
  } else {
    this->Assign(std::move(other).status());
  }
}
template <typename T>
template <typename... Args>
StatusOr<T>::StatusOr(status_impl::in_place_t, Args&&... args)
    : Base(status_impl::in_place, std::forward<Args>(args)...) {}

template <typename T>
template <typename U, typename... Args>
StatusOr<T>::StatusOr(status_impl::in_place_t, std::initializer_list<U> ilist,
                      Args&&... args)
    : Base(status_impl::in_place, ilist, std::forward<Args>(args)...) {}

template <typename T>
const Status& StatusOr<T>::status() const& {
  return this->status_;
}

template <typename T>
Status StatusOr<T>::status() && {
  if (ok()) {
    return OkStatus();
  } else {
    return status_impl::exchange(this->status_, this->status_.code());
  }
}

template <typename T>
const T& StatusOr<T>::value() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::value() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::value() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::value() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
const T& StatusOr<T>::operator*() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::operator*() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::operator*() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::operator*() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
const T* StatusOr<T>::operator->() const {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
T* StatusOr<T>::operator->() {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) const& {
  if (ok()) {
    return this->data_;
  }
  return std::forward<U>(default_value);
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) && {
  if (ok()) {
    return std::move(this->data_);
  }
  return std::forward<U>(default_value);
}

template <typename T>
void StatusOr<T>::IgnoreError() const {
  this->status_.IgnoreError();
}

template <typename T>
IREE_MUST_USE_RESULT static inline bool IsOk(const StatusOr<T>& status_or) {
  return status_or.ok();
}

}  // namespace iree

// Executes an expression `rexpr` that returns a `iree::StatusOr<T>`. On OK,
// moves its value into the variable defined by `lhs`, otherwise returns
// from the current function.
#define IREE_ASSIGN_OR_RETURN(lhs, rexpr)      \
  IREE_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_( \
      IREE_STATUS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, (rexpr))

#define IREE_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(statusor, lhs, rexpr) \
  auto statusor = rexpr;                                                  \
  if (IREE_UNLIKELY(!::iree::IsOk(statusor))) {                           \
    return std::move(statusor).status();                                  \
  }                                                                       \
  lhs = std::move(statusor).value()

#endif  // IREE_BASE_STATUS_CC_H_
