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

#include <string.h>

#include "absl/strings/match.h"
#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/vm/bytecode_module_impl.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

// TODO(benvanik): replace with flatcc version so this file can be pure C.
#include "flatbuffers/flatbuffers.h"
#include "iree/schemas/bytecode_module_def_generated.h"

#define IREE_VM_GET_MODULE_DEF(module)                 \
  ::flatbuffers::GetRoot<iree::vm::BytecodeModuleDef>( \
      module->flatbuffer_data.data)

// Returns true if the given |type_def| is valid, meaning that the type it was
// resolved from is registered or known to the system as a builtin.
static bool iree_vm_type_def_is_valid(iree_vm_type_def_t type_def) {
  return type_def.value_type != IREE_VM_VALUE_TYPE_NONE ||
         type_def.ref_type != IREE_VM_REF_TYPE_NULL;
}

// Resolves a type through either builtin rules or the ref registered types.
static iree_vm_type_def_t iree_vm_bytecode_module_resolve_type(
    const iree::vm::TypeDef* type_def) {
  auto full_name = iree::WrapString(type_def->full_name());
  iree_vm_type_def_t result;
  memset(&result, 0, sizeof(result));
  if (full_name == "i8") {
    result.value_type = IREE_VM_VALUE_TYPE_I8;
  } else if (full_name == "i16") {
    result.value_type = IREE_VM_VALUE_TYPE_I16;
  } else if (full_name == "i32") {
    result.value_type = IREE_VM_VALUE_TYPE_I32;
  } else if (full_name == "i64") {
    result.value_type = IREE_VM_VALUE_TYPE_I64;
  } else if (!full_name.empty() && full_name[0] == '!') {
    full_name.remove_prefix(1);
    if (absl::StartsWith(full_name, "vm.list<")) {
      // This is a !vm.list<...> type. We don't actually care about the type as
      // we allow list types to be widened.
      full_name.remove_suffix(full_name.size() - 7);
    }
    const iree_vm_ref_type_descriptor_t* type_descriptor =
        iree_vm_ref_lookup_registered_type(
            iree_string_view_t{full_name.data(), full_name.size()});
    if (type_descriptor) {
      result.ref_type = type_descriptor->type;
    }
  }
  return result;
}

// Resolves all types through either builtin rules or the ref registered types.
// |type_table| can be omitted to just perform verification that all types are
// registered.
static iree_status_t iree_vm_bytecode_module_resolve_types(
    const iree::vm::BytecodeModuleDef* module_def,
    iree_vm_type_def_t* type_table) {
  for (int i = 0; i < module_def->types()->size(); ++i) {
    type_table[i] =
        iree_vm_bytecode_module_resolve_type(module_def->types()->Get(i));
    if (!iree_vm_type_def_is_valid(type_table[i])) {
      return IREE_STATUS_NOT_FOUND;
    }
  }
  return IREE_STATUS_OK;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_vm_bytecode_module_flatbuffer_verify(
    const iree::vm::BytecodeModuleDef* module_def) {
  if (!module_def->name() || module_def->name()->size() == 0) {
    LOG(ERROR) << "All modules must have a name.";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (!module_def->types()) {
    LOG(ERROR) << "Type table is mandatory, though it could be empty (in empty "
                  "modules).";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (!module_def->exported_functions() ||
      module_def->exported_functions()->size() == 0) {
    LOG(ERROR) << "At least one exported function is required.";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (!module_def->internal_functions() ||
      module_def->internal_functions()->size() == 0) {
    LOG(ERROR) << "At least one internal function is required.";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (!module_def->function_descriptors() ||
      module_def->function_descriptors()->size() !=
          module_def->internal_functions()->size()) {
    LOG(ERROR)
        << "All internal functions need a mapping into the bytecode data.";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (!module_def->bytecode_data()) {
    LOG(ERROR) << "Bytecode data is required if we have any functions.";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  for (int i = 0; i < module_def->types()->size(); ++i) {
    const auto* type_def = module_def->types()->Get(i);
    if (!type_def) {
      LOG(ERROR) << "All types must be valid.";
      return IREE_STATUS_INVALID_ARGUMENT;
    } else if (!type_def->full_name() || type_def->full_name()->size() == 0) {
      LOG(ERROR) << "All types require a name.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    if (!iree_vm_type_def_is_valid(
            iree_vm_bytecode_module_resolve_type(type_def))) {
      LOG(ERROR) << "No type registered with name '"
                 << type_def->full_name()->c_str() << "'.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }
  }

  if (module_def->imported_functions()) {
    for (int i = 0; i < module_def->imported_functions()->size(); ++i) {
      auto* import_def = module_def->imported_functions()->Get(i);
      if (!import_def) {
        LOG(ERROR) << "All imports must be valid.";
        return IREE_STATUS_INVALID_ARGUMENT;
      } else if (!import_def->full_name() ||
                 import_def->full_name()->size() == 0) {
        LOG(ERROR) << "All imports require a name.";
        return IREE_STATUS_INVALID_ARGUMENT;
      } else if (!import_def->signature()) {
        LOG(ERROR) << "All imports require a signature.";
        return IREE_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  for (int i = 0; i < module_def->exported_functions()->size(); ++i) {
    auto* export_def = module_def->exported_functions()->Get(i);
    if (!export_def) {
      LOG(ERROR) << "All exports must be valid.";
      return IREE_STATUS_INVALID_ARGUMENT;
    } else if (!export_def->local_name() ||
               export_def->local_name()->size() == 0) {
      LOG(ERROR) << "All exports require a name.";
      return IREE_STATUS_INVALID_ARGUMENT;
    } else if (!export_def->signature()) {
      LOG(ERROR) << "All exports require a signature.";
      return IREE_STATUS_INVALID_ARGUMENT;
    } else if (export_def->internal_ordinal() < 0 ||
               export_def->internal_ordinal() >=
                   module_def->internal_functions()->size()) {
      LOG(ERROR)
          << "Out-of-bounds reference to a function in the internal table.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }
  }

  for (int i = 0; i < module_def->internal_functions()->size(); ++i) {
    auto* function_def = module_def->internal_functions()->Get(i);
    if (!function_def) {
      LOG(ERROR) << "All functions must be valid.";
      return IREE_STATUS_INVALID_ARGUMENT;
    } else if (!function_def->signature()) {
      LOG(ERROR) << "All functions require a signature.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }

    const auto* function_descriptor =
        module_def->function_descriptors()->Get(i);
    if (!function_descriptor || function_descriptor->bytecode_offset() < 0 ||
        function_descriptor->bytecode_length() < 0 ||
        function_descriptor->bytecode_offset() +
                function_descriptor->bytecode_length() >
            module_def->bytecode_data()->size()) {
      LOG(ERROR) << "Bytecode span must be a valid range.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    if (function_descriptor->i32_register_count() > IREE_I32_REGISTER_COUNT ||
        function_descriptor->ref_register_count() > IREE_REF_REGISTER_COUNT) {
      LOG(ERROR) << "Register counts out of range.";
      return IREE_STATUS_INVALID_ARGUMENT;
    }

    // TODO(benvanik): run bytecode verifier on contents.
  }

  return IREE_STATUS_OK;
}

static void iree_vm_bytecode_module_destroy(void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;

  iree_allocator_free(module->flatbuffer_allocator,
                      (void*)module->flatbuffer_data.data);
  module->flatbuffer_data = {NULL, 0};
  module->flatbuffer_allocator = IREE_ALLOCATOR_NULL;

  iree_allocator_free(module->allocator, module);
}

static iree_string_view_t iree_vm_bytecode_module_name(void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);
  return iree_string_view_t{module_def->name()->data(),
                            module_def->name()->size()};
}

static iree_vm_module_signature_t iree_vm_bytecode_module_signature(
    void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);
  iree_vm_module_signature_t signature;
  signature.import_function_count =
      module_def->imported_functions()
          ? module_def->imported_functions()->size()
          : 0;
  signature.export_function_count = module_def->exported_functions()->size();
  signature.internal_function_count = module_def->internal_functions()->size();
  return signature;
}

static iree_status_t iree_vm_bytecode_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, int32_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (out_function) {
    memset(out_function, 0, sizeof(iree_vm_function_t));
  }
  if (out_name) {
    out_name->data = NULL;
    out_name->size = 0;
  }
  if (out_signature) {
    memset(out_signature, 0, sizeof(iree_vm_function_signature_t));
  }

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);

  const ::flatbuffers::String* name = nullptr;
  const iree::vm::FunctionSignatureDef* signature = nullptr;
  if (linkage == IREE_VM_FUNCTION_LINKAGE_IMPORT) {
    if (!module_def->imported_functions() || ordinal < 0 ||
        ordinal >= module_def->imported_functions()->size()) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    auto* import_def = module_def->imported_functions()->Get(ordinal);
    name = import_def->full_name();
    signature = import_def->signature();
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = linkage;
      out_function->ordinal = ordinal;
    }
  } else if (linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT) {
    if (ordinal < 0 || ordinal >= module_def->exported_functions()->size()) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    auto* export_def = module_def->exported_functions()->Get(ordinal);
    name = export_def->local_name();
    signature = export_def->signature();
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
      out_function->ordinal = export_def->internal_ordinal();

      const iree_vm_function_descriptor_t* function_descriptor =
          &module->function_descriptor_table[export_def->internal_ordinal()];
      out_function->i32_register_count =
          function_descriptor->i32_register_count;
      out_function->ref_register_count =
          function_descriptor->ref_register_count;
    }
  } else {
    if (ordinal < 0 || ordinal >= module_def->internal_functions()->size()) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    auto* function_def = module_def->internal_functions()->Get(ordinal);
    name = function_def->local_name();
    signature = function_def->signature();
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
      out_function->ordinal = ordinal;

      const iree_vm_function_descriptor_t* function_descriptor =
          &module->function_descriptor_table[ordinal];
      out_function->i32_register_count =
          function_descriptor->i32_register_count;
      out_function->ref_register_count =
          function_descriptor->ref_register_count;
    }
  }

  if (out_name && name) {
    out_name->data = name->c_str();
    out_name->size = name->size();
  }
  if (out_signature && signature) {
    out_signature->argument_count =
        signature->argument_types() ? signature->argument_types()->size() : 0;
    out_signature->result_count =
        signature->result_types() ? signature->result_types()->size() : 0;
  }

  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_bytecode_module_get_function_reflection_attr(
    void* self, iree_vm_function_linkage_t linkage, int32_t ordinal,
    int32_t index, iree_string_view_t* key, iree_string_view_t* value) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);

  if (linkage != IREE_VM_FUNCTION_LINKAGE_INTERNAL) {
    iree_vm_function_t internal_function;
    iree_vm_bytecode_module_get_function(self, linkage, ordinal,
                                         &internal_function, NULL, NULL);
    linkage = internal_function.linkage;
    ordinal = internal_function.ordinal;
  }

  if (ordinal < 0 || ordinal >= module_def->internal_functions()->size()) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* export_def = module_def->internal_functions()->Get(ordinal);
  const iree::vm::FunctionSignatureDef* signature = export_def->signature();
  if (!signature->reflection_attrs() || index < 0 ||
      index >= signature->reflection_attrs()->size()) {
    return IREE_STATUS_NOT_FOUND;
  }
  const ::iree::vm::ReflectionAttrDef* attr =
      signature->reflection_attrs()->Get(index);
  const ::flatbuffers::String* attr_key = attr->key();
  const ::flatbuffers::String* attr_value = attr->value();
  if (!attr_key || !attr_value) {
    // Because reflection metadata should not impose any overhead for the
    // non reflection case, we do not eagerly validate in on load -- instead
    // verify it structurally as needed.
    return IREE_STATUS_FAILED_PRECONDITION;
  }

  key->data = attr_key->c_str();
  key->size = attr_key->size();
  value->data = attr_value->c_str();
  value->size = attr_value->size();

  return IREE_STATUS_OK;
}

static bool iree_vm_bytecode_module_compare_str(const flatbuffers::String* lhs,
                                                iree_string_view_t rhs) {
  if (!lhs || lhs->size() != rhs.size) return false;
  return strncmp(lhs->c_str(), rhs.data, rhs.size) == 0;
}

static iree_status_t iree_vm_bytecode_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    iree_vm_function_t* out_function) {
  if (!out_function) return IREE_STATUS_INVALID_ARGUMENT;
  memset(out_function, 0, sizeof(iree_vm_function_t));

  if (!name.data || !name.size) return IREE_STATUS_INVALID_ARGUMENT;

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);

  // NOTE: we could organize imports/exports alphabetically so we could bsearch.
  if (linkage == IREE_VM_FUNCTION_LINKAGE_IMPORT) {
    if (!module_def->imported_functions()) {
      return IREE_STATUS_NOT_FOUND;
    }
    for (int ordinal = 0; ordinal < module_def->imported_functions()->size();
         ++ordinal) {
      auto* import_def = module_def->imported_functions()->Get(ordinal);
      if (iree_vm_bytecode_module_compare_str(import_def->full_name(), name)) {
        return iree_vm_bytecode_module_get_function(self, linkage, ordinal,
                                                    out_function, NULL, NULL);
      }
    }
    return IREE_STATUS_NOT_FOUND;
  } else if (linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT) {
    for (int ordinal = 0; ordinal < module_def->exported_functions()->size();
         ++ordinal) {
      auto* export_def = module_def->exported_functions()->Get(ordinal);
      if (iree_vm_bytecode_module_compare_str(export_def->local_name(), name)) {
        return iree_vm_bytecode_module_get_function(
            self, IREE_VM_FUNCTION_LINKAGE_INTERNAL,
            export_def->internal_ordinal(), out_function, NULL, NULL);
      }
    }
    return IREE_STATUS_NOT_FOUND;
  } else {
    for (int ordinal = 0; ordinal < module_def->internal_functions()->size();
         ++ordinal) {
      auto* function_def = module_def->internal_functions()->Get(ordinal);
      if (iree_vm_bytecode_module_compare_str(function_def->local_name(),
                                              name)) {
        return iree_vm_bytecode_module_get_function(
            self, IREE_VM_FUNCTION_LINKAGE_INTERNAL, ordinal, out_function,
            NULL, NULL);
      }
    }
    return IREE_STATUS_NOT_FOUND;
  }
}

// Lays out the nested tables within a |state| structure.
// Returns the total size of the structure and all tables with padding applied.
// |state| may be null if only the structure size is required for allocation.
static iree_host_size_t iree_vm_bytecode_module_layout_state(
    const iree::vm::BytecodeModuleDef* module_def,
    iree_vm_bytecode_module_state_t* state) {
  int rwdata_storage_capacity =
      module_def->module_state()
          ? module_def->module_state()->global_bytes_capacity()
          : 0;
  int global_ref_count = module_def->module_state()
                             ? module_def->module_state()->global_ref_count()
                             : 0;
  int rodata_ref_count =
      module_def->rodata_segments() ? module_def->rodata_segments()->size() : 0;
  int import_function_count = module_def->imported_functions()
                                  ? module_def->imported_functions()->size()
                                  : 0;

  uint8_t* base_ptr = (uint8_t*)state;
  iree_host_size_t offset =
      iree_align(sizeof(iree_vm_bytecode_module_state_t), 16);

  if (state) {
    state->rwdata_storage = {(base_ptr + offset),
                             (iree_host_size_t)rwdata_storage_capacity};
  }
  offset += iree_align(rwdata_storage_capacity, 16);

  if (state) {
    state->global_ref_count = global_ref_count;
    state->global_ref_table = (iree_vm_ref_t*)(base_ptr + offset);
  }
  offset += iree_align(global_ref_count * sizeof(iree_vm_ref_t), 16);

  if (state) {
    state->rodata_ref_count = rodata_ref_count;
    state->rodata_ref_table = (iree_vm_ro_byte_buffer_t*)(base_ptr + offset);
  }
  offset += iree_align(rodata_ref_count * sizeof(iree_vm_ro_byte_buffer_t), 16);

  if (state) {
    state->import_count = import_function_count;
    state->import_table = (iree_vm_function_t*)(base_ptr + offset);
  }
  offset += iree_align(import_function_count * sizeof(iree_vm_function_t), 16);

  return offset;
}

static iree_status_t iree_vm_bytecode_module_alloc_state(
    void* self, iree_allocator_t allocator,
    iree_vm_module_state_t** out_module_state) {
  if (!out_module_state) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module_state = NULL;

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  auto* module_def = IREE_VM_GET_MODULE_DEF(module);

  // Compute the total size required (with padding) for the state structure.
  iree_host_size_t total_state_struct_size =
      iree_vm_bytecode_module_layout_state(module_def, NULL);

  // Allocate the storage for the structure and all its nested tables.
  iree_vm_bytecode_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, total_state_struct_size,
                                             (void**)&state));
  state->allocator = allocator;

  // Perform layout to get the pointers into the storage for each nested table.
  iree_vm_bytecode_module_layout_state(module_def, state);

  // Setup rodata segments to point directly at the flatbuffer memory.
  for (int i = 0; i < state->rodata_ref_count; ++i) {
    const iree::vm::RodataSegmentDef* segment =
        module_def->rodata_segments()->Get(i);
    iree_vm_ro_byte_buffer_t* ref = &state->rodata_ref_table[i];
    iree_atomic_store(&ref->ref_object.counter, 1);
    ref->data.data = segment->data()->Data();
    ref->data.data_length = segment->data()->size();
  }

  *out_module_state = (iree_vm_module_state_t*)state;
  return IREE_STATUS_OK;
}

static void iree_vm_bytecode_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  if (!module_state) return;

  iree_vm_bytecode_module_state_t* state =
      (iree_vm_bytecode_module_state_t*)module_state;

  // Release remaining global references.
  for (int i = 0; i < state->global_ref_count; ++i) {
    iree_vm_ref_release(&state->global_ref_table[i]);
  }

  iree_allocator_free(state->allocator, module_state);
}

static iree_status_t iree_vm_bytecode_module_resolve_import(
    void* self, iree_vm_module_state_t* module_state, int32_t ordinal,
    iree_vm_function_t function) {
  iree_vm_bytecode_module_state_t* state =
      (iree_vm_bytecode_module_state_t*)module_state;
  if (!state) return IREE_STATUS_INVALID_ARGUMENT;
  if (ordinal < 0 || ordinal >= state->import_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  // TODO(benvanik): verify signature.
  state->import_table[ordinal] = function;
  return IREE_STATUS_OK;
}

static iree_status_t iree_vm_bytecode_module_begin_call(
    void* self, iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // NOTE: any work here adds directly to the invocation time. Avoid doing too
  // much work or touching too many unlikely-to-be-cached structures (such as
  // walking the FlatBuffer, which may cause page faults).

  if (!out_result) return IREE_STATUS_INVALID_ARGUMENT;
  memset(out_result, 0, sizeof(iree_vm_execution_result_t));
  if (!stack) return IREE_STATUS_INVALID_ARGUMENT;

  // Only internal functions store the information needed for execution. We
  // allow exports here as well to make things easier to call externally.
  iree_vm_function_t function = call->function;
  if (function.linkage != IREE_VM_FUNCTION_LINKAGE_INTERNAL) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_get_function(
        self, function.linkage, function.ordinal, &function, NULL, NULL));
  }

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  if (function.ordinal < 0 ||
      function.ordinal >= module->function_descriptor_count) {
    // Invalid function ordinal.
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // Enter function (as this is the initial call).
  // The callee's return will take care of storing the output registers when it
  // actually does return, either immediately or in the future via a resume.
  iree_vm_stack_frame_t* callee_frame = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_stack_function_enter(stack, function, call->argument_registers,
                                   call->result_registers, &callee_frame));

  // Jump into the dispatch routine to execute bytecode until the function
  // either returns (synchronous) or yields (asynchronous).
  return iree_vm_bytecode_dispatch(
      module, (iree_vm_bytecode_module_state_t*)callee_frame->module_state,
      stack, callee_frame, out_result);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_bytecode_module_create(
    iree_const_byte_span_t flatbuffer_data,
    iree_allocator_t flatbuffer_allocator, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  if (!out_module) {
    LOG(ERROR) << "Output module argument not set";
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_module = NULL;

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    LOG(ERROR) << "Flatbuffer data is not present or less than 16 bytes";
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (!iree::vm::BytecodeModuleDefBufferHasIdentifier(
                 flatbuffer_data.data)) {
    LOG(ERROR) << "Flatbuffer data does not have bytecode module identifier";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  const iree::vm::BytecodeModuleDef* module_def =
      ::flatbuffers::GetRoot<iree::vm::BytecodeModuleDef>(flatbuffer_data.data);
  if (!module_def) {
    LOG(ERROR) << "Failed getting root from flatbuffer data";
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_flatbuffer_verify(module_def));

  size_t type_table_size =
      module_def->types()->size() * sizeof(iree_vm_type_def_t);

  iree_vm_bytecode_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(iree_vm_bytecode_module_t) + type_table_size,
      (void**)&module));
  module->allocator = allocator;

  module->function_descriptor_count =
      module_def->function_descriptors()->size();
  module->function_descriptor_table =
      (const iree_vm_function_descriptor_t*)module_def->function_descriptors()
          ->data();
  module->bytecode_data = iree_const_byte_span_t{
      module_def->bytecode_data()->Data(), module_def->bytecode_data()->size()};

  module->flatbuffer_data = flatbuffer_data;
  module->flatbuffer_allocator = flatbuffer_allocator;

  module->type_count = module_def->types()->size();
  module->type_table = (iree_vm_type_def_t*)((uint8_t*)module +
                                             sizeof(iree_vm_bytecode_module_t));
  iree_vm_bytecode_module_resolve_types(module_def, module->type_table);

  iree_vm_module_initialize(&module->interface, module);
  module->interface.destroy = iree_vm_bytecode_module_destroy;
  module->interface.name = iree_vm_bytecode_module_name;
  module->interface.signature = iree_vm_bytecode_module_signature;
  module->interface.get_function = iree_vm_bytecode_module_get_function;
  module->interface.lookup_function = iree_vm_bytecode_module_lookup_function;
  module->interface.alloc_state = iree_vm_bytecode_module_alloc_state;
  module->interface.free_state = iree_vm_bytecode_module_free_state;
  module->interface.resolve_import = iree_vm_bytecode_module_resolve_import;
  module->interface.begin_call = iree_vm_bytecode_module_begin_call;
  module->interface.get_function_reflection_attr =
      iree_vm_bytecode_module_get_function_reflection_attr;

  *out_module = &module->interface;
  return IREE_STATUS_OK;
}
