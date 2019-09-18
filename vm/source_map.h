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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_SOURCE_MAP_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_SOURCE_MAP_H_

#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/optional.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/schemas/module_def_generated.h"
#include "third_party/mlir_edge/iree/schemas/source_map_def_generated.h"

namespace iree {
namespace vm {

class SourceLocation {
 public:
  static bool Equal(const SourceLocation& a, const SourceLocation& b);

  SourceLocation() = default;
  SourceLocation(const SourceMapDef& source_map_def,
                 const FunctionSourceMapDef& function_source_map,
                 int location_ordinal)
      : source_map_def_(&source_map_def),
        function_source_map_(&function_source_map),
        location_ordinal_(location_ordinal) {}

  std::string DebugStringShort() const;

  bool empty() const { return source_map_def_ == nullptr; }

 private:
  const SourceMapDef* source_map_def_ = nullptr;
  const FunctionSourceMapDef* function_source_map_ = nullptr;
  int location_ordinal_ = 0;
};

inline bool operator==(const SourceLocation& a, const SourceLocation& b) {
  return SourceLocation::Equal(a, b);
}

inline bool operator!=(const SourceLocation& a, const SourceLocation& b) {
  return !(a == b);
}

class SourceMap {
 public:
  static SourceMap FromModule(const ModuleDef& module_def);

  SourceMap() = default;
  explicit SourceMap(const SourceMapDef& source_map_def)
      : source_map_def_(&source_map_def) {}

  bool empty() const { return source_map_def_ == nullptr; }
  const SourceMapDef* def() const { return source_map_def_; }

  StatusOr<absl::string_view> GetUniqueString(int string_index) const;

  StatusOr<const FunctionSourceMapDef*> GetFunctionSourceMap(
      int function_ordinal) const;

 private:
  const SourceMapDef* source_map_def_ = nullptr;
};
inline std::ostream& operator<<(std::ostream& stream,
                                const SourceLocation& location) {
  stream << location.DebugStringShort();
  return stream;
}

class SourceMapResolver {
 public:
  static SourceMapResolver FromFunction(const ModuleDef& module_def,
                                        int function_ordinal);

  SourceMapResolver() = default;

  absl::optional<SourceLocation> ResolveBytecodeOffset(int offset) const;

 private:
  SourceMapResolver(SourceMap source_map,
                    const FunctionSourceMapDef& function_source_map)
      : source_map_(std::move(source_map)),
        function_source_map_(&function_source_map) {}

  SourceMap source_map_;
  const FunctionSourceMapDef* function_source_map_ = nullptr;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_SOURCE_MAP_H_
