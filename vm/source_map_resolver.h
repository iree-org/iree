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

#ifndef IREE_VM_SOURCE_MAP_RESOLVER_H_
#define IREE_VM_SOURCE_MAP_RESOLVER_H_

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "base/status.h"
#include "rt/source_resolver.h"
#include "schemas/module_def_generated.h"
#include "schemas/source_map_def_generated.h"

namespace iree {
namespace vm {

class SourceMapResolver final : public rt::SourceResolver {
 public:
  static SourceMapResolver FromModule(const ModuleDef& module_def);

  SourceMapResolver() = default;
  explicit SourceMapResolver(const SourceMapDef& source_map_def)
      : source_map_def_(&source_map_def) {}

  bool empty() const { return source_map_def_ == nullptr; }
  const SourceMapDef* def() const { return source_map_def_; }

  StatusOr<absl::string_view> GetUniqueString(int string_index) const;

  StatusOr<const FunctionSourceMapDef*> GetFunctionSourceMap(
      int function_ordinal) const;

  absl::optional<rt::SourceLocation> ResolveFunctionOffset(
      const rt::Function& function, rt::SourceOffset offset) override;

  void PrintSourceLocation(rt::SourceResolverArgs resolver_args,
                           std::ostream* stream) const override;

 private:
  const SourceMapDef* source_map_def_ = nullptr;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_SOURCE_MAP_RESOLVER_H_
