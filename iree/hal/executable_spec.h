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

#ifndef IREE_HAL_EXECUTABLE_SPEC_H_
#define IREE_HAL_EXECUTABLE_SPEC_H_

#include "absl/types/span.h"
#include "iree/hal/executable_format.h"

namespace iree {
namespace hal {

// Defines an executable specification used by a cache to prepare an executable.
struct ExecutableSpec {
  // TODO(benvanik): pre-populated hash_code/key to avoid calculation.

  // A reference to the executable data as input to the cache.
  // If ExecutableCachingMode::kAliasProvidedData is set then this reference
  // may be retained by the cache and the backing buffer must be kept valid for
  // the lifetime of the cache.
  absl::Span<const uint8_t> executable_data;

  // TODO(benvanik): add specialization info (constants/defines).
  // TODO(benvanik): add compiler flags? could treat as opaque.
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EXECUTABLE_SPEC_H_
