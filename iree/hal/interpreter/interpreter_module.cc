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

#include "iree/hal/interpreter/interpreter_module.h"

#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/interpreter/bytecode_dispatch.h"
#include "iree/hal/interpreter/bytecode_tables_interpreter.h"

namespace iree {
namespace hal {

namespace {

using ::iree::hal::BufferView;

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

}  // namespace

// static
Status InterpreterModule::ValidateStructure(const ModuleDef& module_def) {
  IREE_TRACE_SCOPE0("InterpreterModule::ValidateStructure");

  // Must have a function table.
  if (module_def.function_table()) {
    RETURN_IF_ERROR(
        ValidateFunctionTableStructure(*module_def.function_table()));
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "ModuleDef is missing a function table";
  }

  return OkStatus();
}

// static
Status InterpreterModule::ValidateArgType(const BufferView& arg,
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

// static
StatusOr<ref_ptr<InterpreterModule>> InterpreterModule::FromDef(
    hal::Allocator* allocator, const ModuleDef& module_def) {
  ASSIGN_OR_RETURN(auto module_file, ModuleFile::Create(&module_def, []() {}));
  if (module_file->root() == nullptr) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "No root ModuleDef present";
  }

  auto module =
      assign_ref(new InterpreterModule(allocator, std::move(module_file)));

  // TODO(benvanik): validate internals here? or make explicit?

  return {std::move(module)};
}

InterpreterModule::InterpreterModule(hal::Allocator* allocator,
                                     ref_ptr<ModuleFile> module_file)
    : allocator_(allocator),
      module_file_(std::move(module_file)),
      module_def_(*module_file_->root()) {}

StatusOr<int32_t> InterpreterModule::MapFunctionOrdinal(
    Function::Linkage linkage, int32_t ordinal) const {
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

StatusOr<const Function> InterpreterModule::LookupFunctionByOrdinal(
    Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  return Function(this, Function::Linkage::kInternal, ordinal);
}

StatusOr<const FunctionDef*> InterpreterModule::GetFunctionDef(
    Function::Linkage linkage, int32_t ordinal) const {
  ASSIGN_OR_RETURN(ordinal, MapFunctionOrdinal(linkage, ordinal));
  const auto& function_defs = *function_table_def().functions();
  if (ordinal >= function_defs.size()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Internal function ordinal " << ordinal
           << " out of range of table (" << function_defs.size() << ")";
  }
  return function_defs.Get(ordinal);
}

Status InterpreterModule::Execute(
    Stack* stack, const Function function,
    absl::InlinedVector<hal::BufferView, 8> arguments,
    absl::InlinedVector<hal::BufferView, 8>* results) const {
  IREE_TRACE_SCOPE0("InterperterModule::Execute");

  // Push stack frame for the function we are calling.
  ASSIGN_OR_RETURN(auto* callee_stack_frame, stack->PushFrame(function));

  // TODO(benvanik): rework register storage interface.
  ASSIGN_OR_RETURN(const auto* function_def,
                   GetFunctionDef(function.linkage(), function.ordinal()));
  auto* registers = callee_stack_frame->mutable_registers();
  registers->buffer_views.resize(function_def->bytecode()->local_count());

  // Marshal input arguments.
  for (int i = 0; i < arguments.size(); ++i) {
    registers->buffer_views[i] = std::move(arguments[i]);
  }

  // Run main dispatch loop until it exits (or errors).
  RETURN_IF_ERROR(Dispatch(allocator_, &kernel_runtime_state_, stack,
                           callee_stack_frame, absl::MakeSpan(*results)));

  // Pop the callee frame to balance out the stack.
  RETURN_IF_ERROR(stack->PopFrame());

  return OkStatus();
}

}  // namespace hal
}  // namespace iree
