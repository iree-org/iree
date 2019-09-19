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

#ifndef IREE_VM_EXECUTABLE_TABLE_H_
#define IREE_VM_EXECUTABLE_TABLE_H_

#include "iree/base/status.h"
#include "iree/schemas/executable_table_def_generated.h"

namespace iree {
namespace vm {

// A table of executables present within a module.
// Manages lookup and selection of executables based on target devices.
//
// Thread-safe.
class ExecutableTable {
 public:
  static Status ValidateStructure(
      const ExecutableTableDef& executable_table_def);

  explicit ExecutableTable(const ExecutableTableDef& executable_table_def);
  ExecutableTable(const ExecutableTable&) = delete;
  ExecutableTable& operator=(const ExecutableTable&) = delete;
  ~ExecutableTable();

  const ExecutableTableDef& def() const { return executable_table_def_; }

  StatusOr<const MultiArchExecutableDef*> LookupMultiArchExecutable(
      int executable_ordinal) const;

  // TODO(benvanik): resolve executable by ID+format+features (ExecutableDef).

  // TODO(benvanik): insert/get HAL executables (thread-safe!).

 private:
  const ExecutableTableDef& executable_table_def_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_EXECUTABLE_TABLE_H_
