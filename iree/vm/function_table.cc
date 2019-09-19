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

#include "iree/vm/function_table.h"

#include "absl/container/flat_hash_map.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

namespace {

Status ValidateType(const FunctionTypeDef& type_def) {
  // Ensure all fields are populated.
  return OkStatus();
}

}  // namespace

// static
Status FunctionTable::ValidateStructure(
    const FunctionTableDef& function_table_def) {
  if (!function_table_def.functions()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Function table is missing the function listing";
  }

  // All functions must contain a valid type.
  const auto& functions = *function_table_def.functions();
  for (int i = 0; i < functions.size(); ++i) {
    const auto* function = functions[i];
    if (!function) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Function ordinal " << i << " is missing its contents";
    }
    if (!function->type()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Function ordinal " << i << " is missing its type";
    }
    RETURN_IF_ERROR(ValidateType(*function->type()));
  }

  // Imports must also have a name (that we can use to resolve it).
  if (function_table_def.imports()) {
    const auto& imports = *function_table_def.imports();
    for (int i = 0; i < imports.size(); ++i) {
      int function_index = imports[i];
      if (!functions[function_index]->name()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Import ordinal " << i << " is missing its contents";
      }
    }
  }

  // Exports must also have a name (that others will use to look it up).
  if (function_table_def.exports()) {
    const auto& exports = *function_table_def.exports();
    for (int i = 0; i < exports.size(); ++i) {
      int function_index = exports[i];
      if (!functions[function_index]->name()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Export ordinal " << i << " is missing its contents";
      }
    }
  }

  return OkStatus();
}

FunctionTable::FunctionTable(const Module& module,
                             const FunctionTableDef& function_table_def)
    : module_(module), function_table_def_(function_table_def) {}

FunctionTable::~FunctionTable() = default;

Status FunctionTable::ResolveImports(ImportResolver import_resolver) {
  if (!function_table_def_.imports()) {
    // No imports to resolve.
    return OkStatus();
  }

  const auto& imports = *function_table_def_.imports();
  const auto& functions = *function_table_def_.functions();
  for (int i = 0; i < imports.size(); ++i) {
    const auto* function_def = functions[imports[i]];
    ASSIGN_OR_RETURN(auto import_function,
                     import_resolver(module_, *function_def));
    import_functions_.push_back(std::move(import_function));
  }

  return OkStatus();
}

StatusOr<int> FunctionTable::LookupImportOrdinal(
    absl::string_view import_name) const {
  if (function_table_def_.imports()) {
    const auto& imports = *function_table_def_.imports();
    const auto& functions = *function_table_def_.functions();
    for (int i = 0; i < imports.size(); ++i) {
      if (WrapString(functions[imports[i]]->name()) == import_name) {
        return i;
      }
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Import with the name '" << import_name << "' not found in module";
}

StatusOr<const ImportFunction*> FunctionTable::LookupImport(
    absl::string_view import_name) const {
  ASSIGN_OR_RETURN(int import_ordinal, LookupImportOrdinal(import_name));
  return LookupImport(import_ordinal);
}

StatusOr<const ImportFunction*> FunctionTable::LookupImport(
    int import_ordinal) const {
  if (import_ordinal < 0 || import_ordinal >= import_functions_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Import ordinal " << import_ordinal
           << " is outside the valid range [0, " << import_functions_.size()
           << ")";
  }
  return {&import_functions_[import_ordinal]};
}

StatusOr<int> FunctionTable::LookupExportFunctionOrdinal(
    absl::string_view export_name) const {
  // NOTE: this is a linear scan of the export table, but since export count
  // is usually small and the only time this lookup should happen is on module
  // load it's (probably) fine.
  if (function_table_def_.exports()) {
    const auto& exports = *function_table_def_.exports();
    for (int i = 0; i < exports.size(); ++i) {
      int export_ordinal = exports.Get(i);
      const auto& function_def =
          *function_table_def_.functions()->Get(export_ordinal);
      if (WrapString(function_def.name()) == export_name) {
        return export_ordinal;
      }
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Export with the name '" << export_name << "' not found in module";
}

StatusOr<const Function> FunctionTable::LookupExport(
    absl::string_view export_name) const {
  ASSIGN_OR_RETURN(int export_ordinal,
                   LookupExportFunctionOrdinal(export_name));
  return LookupFunction(export_ordinal);
}

StatusOr<const Function> FunctionTable::LookupExport(int export_ordinal) const {
  if (!function_table_def_.exports() || export_ordinal < 0 ||
      export_ordinal >= function_table_def_.exports()->size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Export ordinal " << export_ordinal
           << " is outside the valid range [0, "
           << function_table_def_.exports()->size() << ")";
  }
  const auto& exports = *function_table_def_.exports();
  int function_ordinal = exports.Get(export_ordinal);
  return LookupFunction(function_ordinal);
}

StatusOr<const Function> FunctionTable::LookupFunction(int ordinal) const {
  if (ordinal < 0 || ordinal >= function_table_def_.functions()->size()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Function ordinal " << ordinal
           << " is outside the valid range [0, "
           << function_table_def_.functions()->size() << ")";
  }
  const auto* function_def = function_table_def_.functions()->Get(ordinal);
  return Function(module_, *function_def);
}

StatusOr<int> FunctionTable::LookupFunctionOrdinal(
    const Function& function) const {
  const auto& functions = *function_table_def_.functions();
  for (int i = 0; i < functions.size(); ++i) {
    if (&function.def() == functions.Get(i)) {
      return i;
    }
  }
  return NotFoundErrorBuilder(IREE_LOC) << "Function not a member of module";
}

StatusOr<int> FunctionTable::LookupFunctionOrdinalByName(
    absl::string_view name) const {
  for (int i = 0; i < function_table_def_.functions()->size(); ++i) {
    const auto* function_def = function_table_def_.functions()->Get(i);
    if (WrapString(function_def->name()) == name) {
      return i;
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Function '" << name
         << "' not found in function table (or names have been stripped)";
}

Status FunctionTable::RegisterBreakpoint(int function_ordinal, int offset,
                                         BreakpointCallback callback) {
  if (breakpoint_tables_.empty()) {
    breakpoint_tables_.resize(function_table_def_.functions()->size());
  }
  if (function_ordinal < 0 || function_ordinal > breakpoint_tables_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Function ordinal " << function_ordinal << " out of bounds";
  }
  if (!breakpoint_tables_[function_ordinal]) {
    breakpoint_tables_[function_ordinal] =
        absl::make_unique<absl::flat_hash_map<int, BreakpointCallback>>();
  }
  auto& function_table = *breakpoint_tables_[function_ordinal];
  function_table[offset] = std::move(callback);
  return OkStatus();
}

Status FunctionTable::UnregisterBreakpoint(int function_ordinal, int offset) {
  if (function_ordinal < 0 || function_ordinal > breakpoint_tables_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Function ordinal " << function_ordinal << " out of bounds";
  }
  auto* function_table = breakpoint_tables_[function_ordinal].get();
  if (function_table) {
    auto it = function_table->find(offset);
    if (it != function_table->end()) {
      function_table->erase(it);
    }
  }
  return OkStatus();
}

Status FunctionTable::UnregisterAllBreakpoints() {
  breakpoint_tables_.clear();
  return OkStatus();
}

FunctionTable::BreakpointTable* FunctionTable::GetFunctionBreakpointTable(
    int function_ordinal) const {
  if (function_ordinal < 0 || function_ordinal >= breakpoint_tables_.size()) {
    return nullptr;
  }
  return breakpoint_tables_[function_ordinal].get();
}

}  // namespace vm
}  // namespace iree
