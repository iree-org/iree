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

#include "iree/vm/source_map_resolver.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/schemas/source_map_def_generated.h"

namespace iree {
namespace vm {

namespace {

Status PrintLocation(const SourceMapResolver& source_map,
                     const FunctionSourceMapDef& function_source_map,
                     const LocationDef& location, std::ostream* stream);

Status PrintFileLocation(const SourceMapResolver& source_map,
                         const FunctionSourceMapDef& function_source_map,
                         const FileLocationDef& location,
                         std::ostream* stream) {
  ASSIGN_OR_RETURN(auto filename,
                   source_map.GetUniqueString(location.filename()));
  *stream << filename << ":" << location.line() << ":" << location.column();
  return OkStatus();
}

Status PrintNameLocation(const SourceMapResolver& source_map,
                         const FunctionSourceMapDef& function_source_map,
                         const NameLocationDef& location,
                         std::ostream* stream) {
  ASSIGN_OR_RETURN(auto name, source_map.GetUniqueString(location.name()));
  *stream << "\"" << name << "\"";
  return OkStatus();
}

Status PrintCallSiteLocation(const SourceMapResolver& source_map,
                             const FunctionSourceMapDef& function_source_map,
                             const CallSiteLocationDef& location,
                             std::ostream* stream) {
  *stream << "(callsites todo)";
  return OkStatus();
}

Status PrintFusedLocation(const SourceMapResolver& source_map,
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

Status PrintLocation(const SourceMapResolver& source_map,
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
      return UnimplementedErrorBuilder(IREE_LOC)
             << "Unhandled location type "
             << static_cast<int>(location.location_union_type());
  }
}

}  // namespace

// static
SourceMapResolver SourceMapResolver::FromModule(const ModuleDef& module_def) {
  if (module_def.source_map()) {
    return SourceMapResolver{*module_def.source_map()};
  }
  return {};
}

StatusOr<absl::string_view> SourceMapResolver::GetUniqueString(
    int string_index) const {
  if (empty()) {
    return NotFoundErrorBuilder(IREE_LOC) << "No source map present";
  }
  const auto* string_table = source_map_def_->string_table();
  if (string_table && string_table->size() > string_index) {
    return WrapString(string_table->Get(string_index));
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "String index " << string_index << " not present in string table";
}

StatusOr<const FunctionSourceMapDef*> SourceMapResolver::GetFunctionSourceMap(
    int function_ordinal) const {
  if (empty()) {
    return NotFoundErrorBuilder(IREE_LOC) << "No source map present";
  }
  const auto* function_table = source_map_def_->function_table();
  if (function_table && function_table->size() > function_ordinal) {
    const auto* function_source_map = function_table->Get(function_ordinal);
    if (function_source_map && function_source_map->location_table() &&
        function_source_map->bytecode_map()) {
      return function_source_map;
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Function ordinal " << function_ordinal
         << " source map not present in function table";
}

absl::optional<rt::SourceLocation> SourceMapResolver::ResolveFunctionOffset(
    const rt::Function& function, rt::SourceOffset offset) {
  if (empty()) return absl::nullopt;
  auto function_source_map_or = GetFunctionSourceMap(function.ordinal());
  if (!function_source_map_or.ok()) {
    return absl::nullopt;
  }
  const auto* function_source_map = function_source_map_or.ValueOrDie();
  const auto* bytecode_map = function_source_map->bytecode_map();
  if (!bytecode_map) return absl::nullopt;

  // TODO(benvanik): allow fuzzy offset matching/table sparsity.
  int location_ordinal = -1;
  for (const auto* map_loc : *bytecode_map) {
    if (map_loc->offset() == offset) {
      location_ordinal = map_loc->location();
      break;
    }
  }
  if (location_ordinal == -1) {
    return absl::nullopt;
  }

  return rt::SourceLocation(this,
                            {
                                reinterpret_cast<uint64_t>(function_source_map),
                                static_cast<uint64_t>(location_ordinal),
                            });
}

void SourceMapResolver::PrintSourceLocation(
    rt::SourceResolverArgs resolver_args, std::ostream* stream) const {
  if (empty()) {
    *stream << "<unknown>";
    return;
  }

  auto* function_source_map =
      reinterpret_cast<FunctionSourceMapDef*>(resolver_args[0]);
  int location_ordinal = static_cast<int>(resolver_args[1]);

  const auto& location =
      *function_source_map->location_table()->Get(location_ordinal);
  auto status = PrintLocation(*this, *function_source_map, location, stream);
  if (!status.ok()) {
    *stream << status;
  }
}

}  // namespace vm
}  // namespace iree
