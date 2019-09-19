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

#ifndef IREE_VM_FUNCTION_TABLE_H_
#define IREE_VM_FUNCTION_TABLE_H_

#include <functional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/schemas/function_table_def_generated.h"
#include "iree/vm/function.h"

namespace iree {
namespace vm {

class Stack;
class Module;

// A table of functions present within a module.
// Manages the import table, local function resolution, and breakpoints.
//
// Function tables are normally thread-compatible. Debugging-specific methods
// like RegisterBreakpoint must only be called when the debugger has suspended
// all fibers that could be executing functions from the table.
class FunctionTable {
 public:
  static Status ValidateStructure(const FunctionTableDef& function_table_def);

  FunctionTable(const Module& module,
                const FunctionTableDef& function_table_def);
  FunctionTable(const FunctionTable&) = delete;
  FunctionTable& operator=(const FunctionTable&) = delete;
  ~FunctionTable();

  const FunctionTableDef& def() const { return function_table_def_; }

  using ImportResolver = std::function<StatusOr<ImportFunction>(
      const Module& importing_module, const FunctionDef& import_function_def)>;
  Status ResolveImports(ImportResolver import_resolver);

  StatusOr<const ImportFunction*> LookupImport(
      absl::string_view import_name) const;
  StatusOr<const ImportFunction*> LookupImport(int import_ordinal) const;

  StatusOr<const Function> LookupExport(absl::string_view export_name) const;
  StatusOr<const Function> LookupExport(int export_ordinal) const;

  StatusOr<const Function> LookupFunction(int ordinal) const;

  StatusOr<int> LookupFunctionOrdinal(const Function& function) const;
  StatusOr<int> LookupFunctionOrdinalByName(absl::string_view name) const;

  // Handles breakpoints that are encountered during execution.
  // The current function and offset within the function will be provided.
  // The fiber is set as suspended prior to issuing the callback and resumed
  // if the callback returns ok.
  //
  // Implementations can use the return status to indicate intended program
  // flow:
  //  - return ok to resume the fiber and continue execution
  //  - return abort to terminate the fiber
  //  - return an error to propagate via normal error handling logic
  using BreakpointCallback = std::function<Status(const Stack& stack)>;

  // Registers a breakpoint for an operation offset within a function.
  // The provided callback will be issued when the breakpoint is hit. If a
  // breakpoint already exists for the given offset it will be replaced.
  //
  // The global debug lock must be held and all fibers must be suspended.
  Status RegisterBreakpoint(int function_ordinal, int offset,
                            BreakpointCallback callback);

  // Unregisters a breakpoint, if one has been registered.
  //
  // The global debug lock must be held and all fibers must be suspended.
  Status UnregisterBreakpoint(int function_ordinal, int offset);

  // Unregisters all breakpoints in the function table.
  //
  // The global debug lock must be held and all fibers must be suspended.
  Status UnregisterAllBreakpoints();

  using BreakpointTable = absl::flat_hash_map<int, BreakpointCallback>;

  // Returns the breakpoint table mapping offset to breakpoint callback.
  // Returns nullptr if the given function does not have a breakpoint table.
  //
  // This table is not synchronized and while the debug lock is held it must not
  // be accessed by any other threads. Reading is otherwise safe.
  BreakpointTable* GetFunctionBreakpointTable(int function_ordinal) const;

 private:
  StatusOr<int> LookupImportOrdinal(absl::string_view import_name) const;
  StatusOr<int> LookupExportFunctionOrdinal(
      absl::string_view export_name) const;

  const Module& module_;
  const FunctionTableDef& function_table_def_;
  std::vector<ImportFunction> import_functions_;

  // One slot per function in the function table. The hash map contains the
  // breakpoints for that particular function mapped by offset within the
  // function.
  std::vector<std::unique_ptr<BreakpointTable>> breakpoint_tables_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_FUNCTION_TABLE_H_
