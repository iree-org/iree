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

#ifndef IREE_RT_SOURCE_LOCATION_H_
#define IREE_RT_SOURCE_LOCATION_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>

namespace iree {
namespace rt {

class SourceResolver;

// An opaque offset into a source map that a SourceResolver can calculate.
// Do not assume that SourceOffset+1 means the next offset as backends are free
// to treat these as everything from pointers to machine code to ordinals in a
// hash table.
using SourceOffset = int64_t;

// Implementation-defined opaque args stored within source locations that can be
// used by a SourceResolver to map back to its internal storage.
using SourceResolverArgs = std::array<uint64_t, 2>;

// A location within a source file.
// Only valid for the lifetime of the SourceResolver that returned it.
class SourceLocation final {
 public:
  // A location where is_unknown always returns true.
  static SourceLocation Unknown() { return SourceLocation(); }

  // Returns true if the two source locations reference the same target.
  inline static bool Equal(const SourceLocation& a, const SourceLocation& b) {
    return a.resolver_ == b.resolver_ && a.resolver_args_ == b.resolver_args_;
  }

  SourceLocation() = default;
  SourceLocation(SourceResolver* resolver, SourceResolverArgs resolver_args)
      : resolver_(resolver), resolver_args_(resolver_args) {}

  // A short one-line human readable string (such as file/line number).
  std::string DebugStringShort() const;

  // Returns true if the source location is unknown, either due to an elided
  // source map entry (such as in optimized builds) or a lack of debugging info
  // for the particular location.
  bool is_unknown() const { return resolver_ == nullptr; }

  // TODO(benvanik): source type (language/format/etc).
  // TODO(benvanik): source file/line.

 private:
  SourceResolver* resolver_ = nullptr;
  SourceResolverArgs resolver_args_ = {0, 0};
};

inline bool operator==(const SourceLocation& a, const SourceLocation& b) {
  return SourceLocation::Equal(a, b);
}

inline bool operator!=(const SourceLocation& a, const SourceLocation& b) {
  return !(a == b);
}

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_SOURCE_LOCATION_H_
