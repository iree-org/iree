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

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include <memory>

#include "iree/base/flatbuffer_util.h"
#include "iree/rt/function.h"
#include "iree/rt/module.h"
#include "iree/schemas/executable_table_def_generated.h"
#include "iree/schemas/function_table_def_generated.h"
#include "iree/schemas/module_def_generated.h"
#include "iree/vm/opcode_info.h"
#include "iree/vm/source_map_resolver.h"

namespace iree {
namespace vm {

using ModuleFile = FlatBufferFile<ModuleDef>;

// A loaded bytecode module backed by a FlatBuffer.
class BytecodeModule : public rt::Module {
 public:
  static Status ValidateStructure(const ModuleDef& module_def);

  ~BytecodeModule() override;

  const ModuleDef& def() const { return module_def_; }
  const FunctionTableDef& function_table_def() const {
    return *module_def_.function_table();
  }
  const ExecutableTableDef& executable_table_def() const {
    return *module_def_.executable_table();
  }

  absl::string_view name() const override {
    return WrapString(module_def_.name());
  }

  const rt::ModuleSignature signature() const override;

  rt::SourceResolver* source_resolver() const override {
    return &source_resolver_;
  }

  rt::Disassembler* disassembler() const override {
    return disassembler_.get();
  }

  std::string DebugStringShort() const override;

  StatusOr<const rt::Function> LookupFunctionByOrdinal(
      rt::Function::Linkage linkage, int32_t ordinal) const override;

  StatusOr<const rt::Function> LookupFunctionByName(
      rt::Function::Linkage linkage, absl::string_view name) const override;

  StatusOr<absl::string_view> GetFunctionName(rt::Function::Linkage linkage,
                                              int32_t ordinal) const override;

  StatusOr<const rt::FunctionSignature> GetFunctionSignature(
      rt::Function::Linkage linkage, int32_t ordinal) const override;

  StatusOr<const FunctionDef*> GetFunctionDef(rt::Function::Linkage linkage,
                                              int32_t ordinal) const;

  StatusOr<const MultiArchExecutableDef*> LookupMultiArchExecutable(
      int executable_ordinal) const;

 protected:
  BytecodeModule(ref_ptr<ModuleFile> module_file, OpcodeTable opcode_table);

  static Status ValidateArgType(const hal::BufferView& arg,
                                const MemRefTypeDef& expected_type);

 private:
  StatusOr<int32_t> MapFunctionOrdinal(rt::Function::Linkage linkage,
                                       int32_t ordinal) const;

  ref_ptr<ModuleFile> module_file_;
  const ModuleDef& module_def_;
  mutable SourceMapResolver source_resolver_;
  mutable std::unique_ptr<rt::Disassembler> disassembler_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_BYTECODE_MODULE_H_
