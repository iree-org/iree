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

#include "iree/vm/executable_table.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"

namespace iree {
namespace vm {

// static
Status ExecutableTable::ValidateStructure(
    const ExecutableTableDef& executable_table_def) {
  if (!executable_table_def.multi_arch_executables()) {
    // May have sequencer only fns. Fine to not have dispatchable executables.
    return OkStatus();
  }

  // All fat executables need at least one device-specific executable.
  const auto& multi_arch_executables =
      *executable_table_def.multi_arch_executables();
  for (int i = 0; i < multi_arch_executables.size(); ++i) {
    const auto* multi_arch_executable = multi_arch_executables[i];
    if (!multi_arch_executable || !multi_arch_executable->executables() ||
        multi_arch_executable->executables()->size() == 0) {
      return InvalidArgumentErrorBuilder(ABSL_LOC)
             << "Multi-arch executable ordinal " << i
             << " is missing its contents";
    }
  }

  return OkStatus();
}

ExecutableTable::ExecutableTable(const ExecutableTableDef& executable_table_def)
    : executable_table_def_(executable_table_def) {}

ExecutableTable::~ExecutableTable() = default;

StatusOr<const MultiArchExecutableDef*>
ExecutableTable::LookupMultiArchExecutable(int executable_ordinal) const {
  if (executable_ordinal < 0 ||
      executable_ordinal >=
          executable_table_def_.multi_arch_executables()->size()) {
    return InvalidArgumentErrorBuilder(ABSL_LOC)
           << "Invalid multi-arch executable ordinal " << executable_ordinal;
  }
  return executable_table_def_.multi_arch_executables()->Get(
      executable_ordinal);
}

}  // namespace vm
}  // namespace iree
