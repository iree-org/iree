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

#ifndef IREE_BASE_INTERNAL_STATUS_WIN32_ERRORS_H_
#define IREE_BASE_INTERNAL_STATUS_WIN32_ERRORS_H_

#include "absl/strings/string_view.h"
#include "iree/base/internal/statusor.h"
#include "iree/base/source_location.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

namespace iree {

// Returns the code for |error| which should be a Win32 error dword.
StatusCode Win32ErrorToCanonicalCode(uint32_t error);

// Returns a StatusBuilder with a status describing the |error| and |location|.
StatusBuilder Win32ErrorToCanonicalStatusBuilder(uint32_t error,
                                                 SourceLocation location);

}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_BASE_INTERNAL_STATUS_WIN32_ERRORS_H_
