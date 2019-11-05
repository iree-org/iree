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

#ifndef IREE_VM_SEQUENCER_MODULE_H_
#define IREE_VM_SEQUENCER_MODULE_H_

#include <memory>

#include "iree/vm/bytecode_module.h"

namespace iree {
namespace vm {

// A module using the sequencer bytecode ops.
class SequencerModule final : public BytecodeModule {
 public:
  static StatusOr<ref_ptr<rt::Module>> FromDef(const ModuleDef& module_def);
  static StatusOr<ref_ptr<rt::Module>> FromFile(
      ref_ptr<ModuleFile> module_file);

  ~SequencerModule() override;

  Status Execute(
      rt::Stack* stack, const rt::Function function,
      absl::InlinedVector<hal::BufferView, 8> arguments,
      absl::InlinedVector<hal::BufferView, 8>* results) const override;

 private:
  explicit SequencerModule(ref_ptr<ModuleFile> module_file);
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_SEQUENCER_MODULE_H_
