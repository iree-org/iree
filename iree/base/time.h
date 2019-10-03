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

#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace iree {

// Converts a relative timeout duration to an absolute deadline time.
// This handles the special cases of absl::ZeroDuration and
// absl::InfiniteDuration to avoid extraneous time queries.
inline absl::Time RelativeTimeoutToDeadline(absl::Duration timeout) {
  if (timeout == absl::ZeroDuration()) {
    return absl::InfinitePast();
  } else if (timeout == absl::InfiniteDuration()) {
    return absl::InfiniteFuture();
  }
  return absl::Now() + timeout;
}

// Suspends execution of the calling thread for the given |duration|.
// Depending on platform this may have an extremely coarse resolution (upwards
// of several to dozens of milliseconds).
inline void Sleep(absl::Duration duration) {
  std::this_thread::sleep_for(absl::ToChronoMilliseconds(duration));
}

}  // namespace iree

#endif  // IREE_BASE_TIME_H_
