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

#include <ctime>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/platform_headers.h"

IREE_API_EXPORT iree_time_t iree_time_now() {
#if defined(IREE_PLATFORM_WINDOWS)
  // GetSystemTimePreciseAsFileTime requires Windows 8, add a fallback
  // (such as using std::chrono) if older support is needed.
  FILETIME system_time;
  ::GetSystemTimePreciseAsFileTime(&system_time);

  constexpr int64_t kUnixEpochStartTicks = 116444736000000000i64;
  constexpr int64_t kFtToMicroSec = 10;
  LARGE_INTEGER li;
  li.LowPart = system_time.dwLowDateTime;
  li.HighPart = system_time.dwHighDateTime;
  li.QuadPart -= kUnixEpochStartTicks;
  li.QuadPart /= kFtToMicroSec;
  return li.QuadPart;
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)
  timespec clock_time;
  clock_gettime(CLOCK_REALTIME, &clock_time);
  return clock_time.tv_nsec;
#else
#error "IREE system clock needs to be set up for your platform"
#endif
}
