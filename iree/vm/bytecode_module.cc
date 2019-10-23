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

#include "iree/vm/bytecode_module.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/buffer_view.h"
#include "iree/vm/bytecode_disassembler.h"

namespace iree {
namespace vm {

namespace {

using ::iree::hal::BufferView;
using ::iree::rt::Function;
using ::iree::rt::FunctionSignature;
using ::iree::rt::Module;
using ::iree::rt::ModuleSignature;

Status ValidateElementSize(int element_bit_width,
                           const ElementTypeDef& expected_element_type) {
  switch (expected_element_type.type_union_type()) {
    case ElementTypeDefUnion::FloatTypeDef: {
      auto expected_bit_width =
          expected_element_type.type_union_as_FloatTypeDef()->width();
      if (element_bit_width != expected_bit_width) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Has element bit width " << element_bit_width
               << " but expected " << expected_bit_width;
      }
      return OkStatus();
    }
    case ElementTypeDefUnion::IntegerTypeDef: {
      auto expected_bit_width =
          expected_element_type.type_union_as_IntegerTypeDef()->width();
      if (element_bit_width != expected_bit_width) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Has element bit width " << element_bit_width
               << " but expected " << expected_bit_width;
      }
      return OkStatus();
    }
    case ElementTypeDefUnion::UnknownTypeDef:
    case ElementTypeDefUnion::NONE: {
    }
  }
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "Defined type has unsupported element type "
         << EnumNameElementTypeDefUnion(
                expected_element_type.type_union_type());
}

Status ValidateTypeStructure(const FunctionTypeDef& type_def) {
  // Ensure all fields are populated.
  return OkStatus();
}

Status ValidateFunctionTableStructure(
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
    RETURN_IF_ERROR(ValidateTypeStructure(*function->type()));
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

Status ValidateExecutableTableStructure(
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
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Multi-arch executable ordinal " << i
             << " is missing its contents";
    }
  }

  return OkStatus();
}

}  // namespace

// static
Status BytecodeModule::ValidateStructure(const ModuleDef& module_def) {
  IREE_TRACE_SCOPE0("BytecodeModule::ValidateStructure");

  // Must have a function table.
  if (module_def.function_table()) {
    RETURN_IF_ERROR(
        ValidateFunctionTableStructure(*module_def.function_table()));
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "ModuleDef is missing a function table";
  }

  // Must have an executable table.
  if (module_def.executable_table()) {
    RETURN_IF_ERROR(
        ValidateExecutableTableStructure(*module_def.executable_table()));
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "ModuleDef is missing an executable table";
  }

  return OkStatus();
}

BytecodeModule::BytecodeModule(std::unique_ptr<ModuleFile> module_file,
                               OpcodeTable opcode_table)
    : module_file_(std::move(module_file)),
      module_def_(*module_file_->root()),
      source_resolver_(SourceMapResolver::FromModule(module_def_)),
      disassembler_(absl::make_unique<BytecodeDisassembler>(opcode_table)) {}

BytecodeModule::~BytecodeModule() = default;

const ModuleSignature BytecodeModule::signature() const {
  return ModuleSignature(function_table_def().imports()->size(),
                         function_table_def().exports()->size(),
                         function_table_def().functions()->size(), 0);
}

std::string BytecodeModule::DebugStringShort() const {
  return std::string(name());
}

StatusOr<int32_t> BytecodeModule::MapFunctionOrdinal(Function::Linkage linkage,
                                                     int32_t ordinal) const {
  const auto& function_table = function_table_def();
  switch (linkage) {
    case Function::Linkage::kImport:
      if (ordinal < 0 || ordinal >= function_table.imports()->size()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Import ordinal " << ordinal
               << " is outside the valid range [0, "
               << function_table.imports()->size() << ")";
      }
      ordinal = function_table.imports()->Get(ordinal);
      break;
    case Function::Linkage::kExport:
      if (ordinal < 0 || ordinal >= function_table.exports()->size()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Export ordinal " << ordinal
               << " is outside the valid range [0, "
               << function_table.exports()->size() << ")";
      }
      ordinal = function_table.exports()->Get(ordinal);
      break;
    default:
      break;
  }
  if (ordinal < 0 || ordinal >= function_table.functions()->size()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Function ordinal " << ordinal
           << " is outside the valid range [0, "
           << function_table.functions()->size() << ")";
  }
  return ordinal;
}

StatusOr<const Function> BytecodeModule::LookupFunctionByOrdinal(
    Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  return Function(this, Function::Linkage::kInternal, ordinal);
}

StatusOr<const Function> BytecodeModule::LookupFunctionByName(
    Function::Linkage linkage, absl::string_view name) const {
  const auto& functions = *function_table_def().functions();
  for (int i = 0; i < functions.size(); ++i) {
    const auto* function_def = functions.Get(i);
    if (WrapString(function_def->name()) == name) {
      return LookupFunctionByOrdinal(Function::Linkage::kInternal, i);
    }
  }
  return NotFoundErrorBuilder(IREE_LOC)
         << "Function '" << name
         << "' not found in function table (or names have been stripped)";
}

StatusOr<absl::string_view> BytecodeModule::GetFunctionName(
    Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  const auto* function_def = function_table_def().functions()->Get(ordinal);
  return WrapString(function_def->name());
}

StatusOr<const FunctionSignature> BytecodeModule::GetFunctionSignature(
    Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  const auto* function_def = function_table_def().functions()->Get(ordinal);
  const auto* type_def = function_def->type();
  return FunctionSignature(
      type_def->inputs() ? type_def->inputs()->size() : 0,
      type_def->results() ? type_def->results()->size() : 0);
}

StatusOr<const FunctionDef*> BytecodeModule::GetFunctionDef(
    rt::Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  const auto& function_defs = *function_table_def().functions();
  if (ordinal >= function_defs.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Internal function ordinal " << ordinal
           << " out of range of table (" << function_defs.size() << ")";
  }
  return function_defs.Get(ordinal);
}

StatusOr<const MultiArchExecutableDef*>
BytecodeModule::LookupMultiArchExecutable(int executable_ordinal) const {
  if (executable_ordinal < 0 ||
      executable_ordinal >=
          executable_table_def().multi_arch_executables()->size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid multi-arch executable ordinal " << executable_ordinal;
  }
  return executable_table_def().multi_arch_executables()->Get(
      executable_ordinal);
}

// static
Status BytecodeModule::ValidateArgType(const BufferView& arg,
                                       const MemRefTypeDef& expected_type) {
  RETURN_IF_ERROR(
      ValidateElementSize(arg.element_size * 8, *expected_type.element_type()));

  auto expected_shape = expected_type.shape();
  if (arg.shape.size() != expected_shape->size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Argument should have rank " << expected_shape->size()
           << " but has rank " << arg.shape.size();
  }
  for (int i = 0; i < expected_shape->size(); ++i) {
    auto dim_size = arg.shape[i];
    auto expected_dim_size = expected_shape->Get(i);
    if (dim_size != expected_dim_size) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Argument dimension " << i << " should have size "
             << expected_dim_size << " but has size " << dim_size;
    }
  }
  return OkStatus();
}

}  // namespace vm
}  // namespace iree
