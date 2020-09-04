// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_METAL_APPLE_TIME_UTIL_H_
#define IREE_HAL_METAL_APPLE_TIME_UTIL_H_

#include <dispatch/dispatch.h>

#include "iree/base/time.h"

namespace iree {
namespace hal {
namespace metal {

// Converts a relative iree::Duration against the currrent time to the
// corresponding dispatch_time_t value.
static inline dispatch_time_t DurationToDispatchTime(Duration duration_ns) {
  if (duration_ns == InfiniteDuration()) return DISPATCH_TIME_FOREVER;
  if (duration_ns == ZeroDuration()) return DISPATCH_TIME_NOW;
  return dispatch_time(DISPATCH_TIME_NOW, static_cast<uint64_t>(duration_ns));
}

// Converts an absolute iree::Time time to the corresponding dispatch_time_t
// value.
static inline dispatch_time_t DeadlineToDispatchTime(Time deadline_ns) {
  return DurationToDispatchTime(DeadlineToRelativeTimeoutNanos(deadline_ns));
}

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_APPLE_TIME_UTIL_H_
