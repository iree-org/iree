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

#ifndef IREE_BASE_TIME_H_
#define IREE_BASE_TIME_H_

#include <type_traits>
#include <utility>

#include "iree/base/api.h"

namespace iree {
namespace impl {
template <class Tag, typename T>
class ChronoType {
 public:
  ChronoType() : value_() {}
  explicit ChronoType(const T& value) : value_(value) {}
  explicit ChronoType(T&& value) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : value_(std::move(value)) {}

  explicit operator T&() noexcept { return value_; }
  explicit operator const T&() const noexcept { return value_; }

  friend void swap(ChronoType& a, ChronoType& b) noexcept {
    using std::swap;
    swap(static_cast<T&>(a), static_cast<T&>(b));
  }

  friend inline bool operator==(const ChronoType& lhs, const ChronoType& rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend inline bool operator!=(const ChronoType& lhs, const ChronoType& rhs) {
    return !(lhs == rhs);
  }
  friend inline bool operator<(const ChronoType& lhs, const ChronoType& rhs) {
    return lhs.value_ < rhs.value_;
  }
  friend inline bool operator>(const ChronoType& lhs, const ChronoType& rhs) {
    return rhs < lhs;
  }
  friend inline bool operator<=(const ChronoType& lhs, const ChronoType& rhs) {
    return !(lhs > rhs);
  }
  friend inline bool operator>=(const ChronoType& lhs, const ChronoType& rhs) {
    return !(lhs < rhs);
  }

  friend ChronoType& operator+=(ChronoType& lhs, const ChronoType& rhs) {
    static_cast<T&>(lhs) += static_cast<const T&>(rhs);
    return lhs;
  }
  friend ChronoType operator+(const ChronoType& lhs, const ChronoType& rhs) {
    return ChronoType(static_cast<const T&>(lhs) + static_cast<const T&>(rhs));
  }

  friend ChronoType& operator-=(ChronoType& lhs, const ChronoType& rhs) {
    static_cast<T&>(lhs) -= static_cast<const T&>(rhs);
    return lhs;
  }
  friend ChronoType operator-(const ChronoType& lhs, const ChronoType& rhs) {
    return ChronoType(static_cast<const T&>(lhs) - static_cast<const T&>(rhs));
  }

 private:
  T value_;
};
}  // namespace impl

struct Duration : public impl::ChronoType<Duration, iree_duration_t> {
  using ChronoType::ChronoType;
  explicit operator uint64_t() const noexcept {
    if (static_cast<iree_duration_t>(*this) == IREE_DURATION_INFINITE) {
      return UINT64_MAX;
    }
    int64_t relative_ns = static_cast<int64_t>(*this);
    return relative_ns <= 0 ? 0 : static_cast<uint64_t>(relative_ns);
  }
};

static inline Duration InfiniteDuration() {
  return Duration(IREE_DURATION_INFINITE);
}
static inline Duration ZeroDuration() { return Duration(IREE_DURATION_ZERO); }

struct Time : public impl::ChronoType<Time, iree_time_t> {
  using ChronoType::ChronoType;
  friend Duration operator+(const Time& lhs, const Time& rhs) {
    if (static_cast<iree_time_t>(lhs) == IREE_TIME_INFINITE_FUTURE ||
        static_cast<iree_time_t>(rhs) == IREE_TIME_INFINITE_FUTURE) {
      return InfiniteDuration();
    } else if (static_cast<iree_time_t>(lhs) == IREE_TIME_INFINITE_PAST ||
               static_cast<iree_time_t>(rhs) == IREE_TIME_INFINITE_PAST) {
      return ZeroDuration();
    }
    return Duration(static_cast<const iree_time_t&>(lhs) +
                    static_cast<const iree_time_t&>(rhs));
  }
  friend Duration operator-(const Time& lhs, const Time& rhs) {
    if (static_cast<iree_time_t>(lhs) == IREE_TIME_INFINITE_FUTURE ||
        static_cast<iree_time_t>(rhs) == IREE_TIME_INFINITE_FUTURE) {
      return InfiniteDuration();
    } else if (static_cast<iree_time_t>(lhs) == IREE_TIME_INFINITE_PAST ||
               static_cast<iree_time_t>(rhs) == IREE_TIME_INFINITE_PAST) {
      return ZeroDuration();
    }
    return Duration(static_cast<const iree_time_t&>(lhs) -
                    static_cast<const iree_time_t&>(rhs));
  }
};

static inline Time InfinitePast() { return Time(IREE_TIME_INFINITE_PAST); }
static inline Time InfiniteFuture() { return Time(IREE_TIME_INFINITE_FUTURE); }

static inline Duration Milliseconds(int64_t millis) {
  return Duration(millis * 1000000ull);
}

// Returns the current system time in unix nanoseconds.
// Depending on the system architecture and power mode this time may have a
// very coarse granularity (on the order of microseconds to milliseconds).
//
// The system timer may not be monotonic; users should ensure when comparing
// times they check for negative values in case the time moves backwards.
static inline Time Now() { return Time(iree_time_now()); }

// Converts a relative timeout duration to an absolute deadline time.
// This handles the special cases of IREE_DURATION_ZERO and
// IREE_DURATION_INFINITE to avoid extraneous time queries.
static inline Time RelativeTimeoutToDeadlineNanos(Duration timeout_ns) {
  return Time(iree_relative_timeout_to_deadline_ns(
      static_cast<iree_duration_t>(timeout_ns)));
}

static inline Duration DeadlineToRelativeTimeoutNanos(Time deadline_ns) {
  if (deadline_ns == InfiniteFuture()) {
    return InfiniteDuration();
  } else if (deadline_ns == InfinitePast()) {
    return ZeroDuration();
  } else {
    return Duration(static_cast<uint64_t>(deadline_ns - Now()));
  }
}

}  // namespace iree

#endif  // IREE_BASE_TIME_H_
