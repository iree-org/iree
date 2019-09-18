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

#include "third_party/mlir_edge/iree/vm/source_map.h"

#include <sstream>

#include "third_party/mlir_edge/iree/base/flatbuffer_util.h"
#include "third_party/mlir_edge/iree/base/status.h"

namespace iree {
namespace vm {

namespace {

Status PrintLocation(const SourceMap& source_map,
                     const FunctionSourceMapDef& function_source_map,
                     const LocationDef& location, std::ostream* stream);

Status PrintFileLocation(const SourceMap& source_map,
                         const FunctionSourceMapDef& function_source_map,
                         const FileLocationDef& location,
                         std::ostream* stream) {
  ASSIGN_OR_RETURN(auto filename,
                   source_map.GetUniqueString(location.filename()));
  *stream << filename << ":" << location.line() << ":" << location.column();
  return OkStatus();
}

Status PrintNameLocation(const SourceMap& source_map,
                         const FunctionSourceMapDef& function_source_map,
                         const NameLocationDef& location,
                         std::ostream* stream) {
  ASSIGN_OR_RETURN(auto name, source_map.GetUniqueString(location.name()));
  *stream << "\"" << name << "\"";
  return OkStatus();
}

Status PrintCallSiteLocation(const SourceMap& source_map,
                             const FunctionSourceMapDef& function_source_map,
                             const CallSiteLocationDef& location,
                             std::ostream* stream) {
  *stream << "(callsites todo)";
  return OkStatus();
}

Status PrintFusedLocation(const SourceMap& source_map,
                          const FunctionSourceMapDef& function_source_map,
                          const FusedLocationDef& location,
                          std::ostream* stream) {
  *stream << "fused[";
  if (location.locations()) {
    for (int i = 0; i < location.locations()->size(); ++i) {
      if (i > 0) *stream << ", ";
      int location_ordinal = location.locations()->Get(i);
      const auto& child_location =
          *function_source_map.location_table()->Get(location_ordinal);
      RETURN_IF_ERROR(PrintLocation(source_map, function_source_map,
                                    child_location, stream));
    }
  }
  *stream << "]";
  return OkStatus();
}

Status PrintLocation(const SourceMap& source_map,
                     const FunctionSourceMapDef& function_source_map,
                     const LocationDef& location, std::ostream* stream) {
  switch (location.location_union_type()) {
    case LocationDefUnion::FileLocationDef:
      return PrintFileLocation(source_map, function_source_map,
                               *location.location_union_as_FileLocationDef(),
                               stream);
    case LocationDefUnion::NameLocationDef:
      return PrintNameLocation(source_map, function_source_map,
                               *location.location_union_as_NameLocationDef(),
                               stream);
    case LocationDefUnion::CallSiteLocationDef:
      return PrintCallSiteLocation(
          source_map, function_source_map,
          *location.location_union_as_CallSiteLocationDef(), stream);
    case LocationDefUnion::FusedLocationDef:
      return PrintFusedLocation(source_map, function_source_map,
                                *location.location_union_as_FusedLocationDef(),
                                stream);
    default:
      return UnimplementedErrorBuilder(ABSL_LOC)
             << "Unhandled location type "
             << static_cast<int>(location.location_union_type());
  }
}

}  // namespace

// static
bool SourceLocation::Equal(const SourceLocation& a, const SourceLocation& b) {
  return a.source_map_def_ == b.source_map_def_ &&
         a.function_source_map_ == b.function_source_map_ &&
         a.location_ordinal_ == b.location_ordinal_;
}

std::string SourceLocation::DebugStringShort() const {
  if (empty()) {
    return "<unknown>";
  }
  std::ostringstream stream;
  const auto& location =
      *function_source_map_->location_table()->Get(location_ordinal_);
  auto status = PrintLocation(SourceMap(*source_map_def_),
                              *function_source_map_, location, &stream);
  if (!status.ok()) {
    stream << status;
  }
  return stream.str();
}

// static
SourceMap SourceMap::FromModule(const ModuleDef& module_def) {
  if (module_def.source_map()) {
    return SourceMap{*module_def.source_map()};
  }
  return {};
}

StatusOr<absl::string_view> SourceMap::GetUniqueString(int string_index) const {
  if (empty()) {
    return NotFoundErrorBuilder(ABSL_LOC) << "No source map present";
  }
  const auto* string_table = source_map_def_->string_table();
  if (string_table && string_table->size() > string_index) {
    return WrapString(string_table->Get(string_index));
  }
  return NotFoundErrorBuilder(ABSL_LOC)
         << "String index " << string_index << " not present in string table";
}

StatusOr<const FunctionSourceMapDef*> SourceMap::GetFunctionSourceMap(
    int function_ordinal) const {
  if (empty()) {
    return NotFoundErrorBuilder(ABSL_LOC) << "No source map present";
  }
  const auto* function_table = source_map_def_->function_table();
  if (function_table && function_table->size() > function_ordinal) {
    const auto* function_source_map = function_table->Get(function_ordinal);
    if (function_source_map && function_source_map->location_table() &&
        function_source_map->bytecode_map()) {
      return function_source_map;
    }
  }
  return NotFoundErrorBuilder(ABSL_LOC)
         << "Function ordinal " << function_ordinal
         << " source map not present in function table";
}

// static
SourceMapResolver SourceMapResolver::FromFunction(const ModuleDef& module_def,
                                                  int function_ordinal) {
  auto source_map = SourceMap::FromModule(module_def);
  if (source_map.empty()) {
    return {};
  }
  auto function_source_map_or =
      source_map.GetFunctionSourceMap(function_ordinal);
  if (!function_source_map_or.ok()) {
    return {};
  }
  return SourceMapResolver(source_map, *function_source_map_or.ValueOrDie());
}

absl::optional<SourceLocation> SourceMapResolver::ResolveBytecodeOffset(
    int offset) const {
  if (!function_source_map_) {
    return {};
  }

  const auto* bytecode_map = function_source_map_->bytecode_map();

  // TODO(benvanik): allow fuzzy offset matching/table sparsity.
  int location_ordinal = -1;
  for (const auto* map_loc : *bytecode_map) {
    if (map_loc->offset() == offset) {
      location_ordinal = map_loc->location();
      break;
    }
  }
  if (location_ordinal == -1) {
    return {};
  }

  return SourceLocation(*source_map_.def(), *function_source_map_,
                        location_ordinal);
}

}  // namespace vm
}  // namespace iree
