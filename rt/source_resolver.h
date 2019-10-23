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

#ifndef IREE_RT_SOURCE_RESOLVER_H_
#define IREE_RT_SOURCE_RESOLVER_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/status.h"
#include "rt/function.h"
#include "rt/source_location.h"

namespace iree {
namespace rt {

// Resolves offsets within functions to SourceLocations and provides source
// language services.
//
// Thread-safe.
class SourceResolver {
 public:
  virtual ~SourceResolver() = default;

  // Resolves a function-relative offset to a source location.
  // Not all offsets within a function may have source mapping information.
  virtual absl::optional<SourceLocation> ResolveFunctionOffset(
      const Function& function, SourceOffset offset) = 0;

  // Converts a source location to a human-readable string, commonly in a single
  // line denoting an original source file location (such as path:line:col).
  virtual void PrintSourceLocation(SourceResolverArgs resolver_args,
                                   std::ostream* stream) const = 0;

  // TODO(benvanik): query local variable names.

  // TODO(benvanik): step target calculation (relative mapping).
  // TODO(benvanik): step target based on SourceLocation delta.

  // TODO(benvanik): expression evaluator? (setting variables)

 protected:
  friend class SourceLocation;

  SourceResolver() = default;

  // TODO(benvanik): get line mapping information.
};

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_SOURCE_RESOLVER_H_
