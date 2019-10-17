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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
#define IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_

#include <string>

#include "iree/base/flatbuffer_util.h"
#include "iree/bindings/python/pyiree/binding.h"
#include "iree/schemas/module_def_generated.h"

namespace iree {
namespace python {

class MemoryModuleFile : public std::enable_shared_from_this<MemoryModuleFile> {
 public:
  MemoryModuleFile() = default;
  explicit MemoryModuleFile(std::unique_ptr<FlatBufferFile<ModuleDef>> file)
      : file_(std::move(file)) {}
  virtual ~MemoryModuleFile() = default;

  FlatBufferFile<ModuleDef>* file() const { return file_.get(); }

 private:
  std::unique_ptr<FlatBufferFile<ModuleDef>> file_;
};

std::shared_ptr<MemoryModuleFile> CompileModuleFromAsm(
    const std::string& moduleAsm);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
