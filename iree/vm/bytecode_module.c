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

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/vm/bytecode_module_impl.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

// Perform an strcmp between a flatbuffers string and an IREE string view.
static bool iree_vm_flatbuffer_strcmp(flatbuffers_string_t lhs,
                                      iree_string_view_t rhs) {
  size_t lhs_size = flatbuffers_string_len(lhs);
  int x = strncmp(lhs, rhs.data, lhs_size < rhs.size ? lhs_size : rhs.size);
  return x != 0 ? x : lhs_size < rhs.size ? -1 : lhs_size > rhs.size;
}

// Returns true if the given |type_def| is valid, meaning that the type it was
// resolved from is registered or known to the system as a builtin.
static bool iree_vm_type_def_is_valid(iree_vm_type_def_t type_def) {
  return type_def.value_type != IREE_VM_VALUE_TYPE_NONE ||
         type_def.ref_type != IREE_VM_REF_TYPE_NULL;
}

// Resolves a type through either builtin rules or the ref registered types.
static iree_vm_type_def_t iree_vm_bytecode_module_resolve_type(
    iree_vm_TypeDef_table_t type_def) {
  iree_vm_type_def_t result;
  memset(&result, 0, sizeof(result));
  flatbuffers_string_t full_name = iree_vm_TypeDef_full_name(type_def);
  if (!flatbuffers_string_len(full_name)) {
    return result;
  } else if (iree_vm_flatbuffer_strcmp(full_name,
                                       iree_make_cstring_view("i8")) == 0) {
    result.value_type = IREE_VM_VALUE_TYPE_I8;
  } else if (iree_vm_flatbuffer_strcmp(full_name,
                                       iree_make_cstring_view("i16")) == 0) {
    result.value_type = IREE_VM_VALUE_TYPE_I16;
  } else if (iree_vm_flatbuffer_strcmp(full_name,
                                       iree_make_cstring_view("i32")) == 0) {
    result.value_type = IREE_VM_VALUE_TYPE_I32;
  } else if (iree_vm_flatbuffer_strcmp(full_name,
                                       iree_make_cstring_view("i64")) == 0) {
    result.value_type = IREE_VM_VALUE_TYPE_I64;
  } else if (full_name[0] == '!') {
    // Note that we drop the ! prefix:
    iree_string_view_t type_name = {full_name + 1,
                                    flatbuffers_string_len(full_name) - 1};
    if (strncmp(type_name.data, "vm.list<", strlen("vm.list<")) == 0) {
      // This is a !vm.list<...> type. We don't actually care about the type as
      // we allow list types to be widened. Rewrite to just vm.list as that's
      // all we have registered.
      type_name = iree_make_cstring_view("vm.list");
    }
    const iree_vm_ref_type_descriptor_t* type_descriptor =
        iree_vm_ref_lookup_registered_type(type_name);
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
    iree_vm_TypeDef_vec_t type_defs, iree_vm_type_def_t* type_table) {
  for (size_t i = 0; i < iree_vm_TypeDef_vec_len(type_defs); ++i) {
    iree_vm_TypeDef_table_t type_def = iree_vm_TypeDef_vec_at(type_defs, i);
    type_table[i] = iree_vm_bytecode_module_resolve_type(type_def);
    if (!iree_vm_type_def_is_valid(type_table[i])) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no type registered with name '%s'",
                              iree_vm_TypeDef_full_name(type_def));
    }
  }
  return iree_ok_status();
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_vm_bytecode_module_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_vm_BytecodeModuleDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_vm_BytecodeModuleDef_table_t module_def =
      iree_vm_BytecodeModuleDef_as_root(flatbuffer_data.data);

  flatbuffers_string_t name = iree_vm_BytecodeModuleDef_name(module_def);
  if (!flatbuffers_string_len(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module missing name field");
  }

  iree_vm_TypeDef_vec_t types = iree_vm_BytecodeModuleDef_types(module_def);
  for (size_t i = 0; i < iree_vm_TypeDef_vec_len(types); ++i) {
    iree_vm_TypeDef_table_t type_def = iree_vm_TypeDef_vec_at(types, i);
    if (!type_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "types[%zu] missing body", i);
    }
    flatbuffers_string_t full_name = iree_vm_TypeDef_full_name(type_def);
    if (flatbuffers_string_len(full_name) <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "types[%zu] missing name", i);
    }
  }

  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  iree_vm_InternalFunctionDef_vec_t internal_functions =
      iree_vm_BytecodeModuleDef_internal_functions(module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);

  if (flatbuffers_vec_len(internal_functions) !=
      flatbuffers_vec_len(function_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mismatched internal_functions/function_descriptors vectors (%zu != "
        "%zu)",
        flatbuffers_vec_len(internal_functions),
        flatbuffers_vec_len(function_descriptors));
  }

  for (size_t i = 0; i < iree_vm_ImportFunctionDef_vec_len(imported_functions);
       ++i) {
    iree_vm_ImportFunctionDef_table_t import_def =
        iree_vm_ImportFunctionDef_vec_at(imported_functions, i);
    if (!import_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "imports[%zu] missing body", i);
    }
    flatbuffers_string_t full_name =
        iree_vm_ImportFunctionDef_full_name(import_def);
    if (!flatbuffers_string_len(full_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "imports[%zu] missing full_name", i);
    }
    if (!iree_vm_ImportFunctionDef_signature(import_def)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "imports[%zu] missing signature", i);
    }
  }

  for (size_t i = 0; i < iree_vm_ExportFunctionDef_vec_len(exported_functions);
       ++i) {
    iree_vm_ExportFunctionDef_table_t export_def =
        iree_vm_ExportFunctionDef_vec_at(exported_functions, i);
    if (!export_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%zu] missing body", i);
    }
    flatbuffers_string_t local_name =
        iree_vm_ExportFunctionDef_local_name(export_def);
    if (!flatbuffers_string_len(local_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%zu] missing local_name", i);
    }
    if (!iree_vm_ExportFunctionDef_signature(export_def)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%zu] missing signature", i);
    }
    iree_host_size_t internal_ordinal =
        iree_vm_ExportFunctionDef_internal_ordinal(export_def);
    if (internal_ordinal >=
        iree_vm_InternalFunctionDef_vec_len(internal_functions)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%zu] internal_ordinal out of bounds (0 < %zu < %zu)", i,
          internal_ordinal,
          iree_vm_InternalFunctionDef_vec_len(internal_functions));
    }
  }

  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  for (size_t i = 0;
       i < iree_vm_InternalFunctionDef_vec_len(internal_functions); ++i) {
    iree_vm_InternalFunctionDef_table_t function_def =
        iree_vm_InternalFunctionDef_vec_at(internal_functions, i);
    if (!function_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "functions[%zu] missing body", i);
    }
    if (!iree_vm_InternalFunctionDef_signature(function_def)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "functions[%zu] missing signature", i);
    }

    iree_vm_FunctionDescriptor_struct_t function_descriptor =
        iree_vm_FunctionDescriptor_vec_at(function_descriptors, i);
    if (function_descriptor->bytecode_offset < 0 ||
        function_descriptor->bytecode_offset +
                function_descriptor->bytecode_length >
            flatbuffers_uint8_vec_len(bytecode_data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "functions[%zu] descriptor bytecode span out of range (0 < %d < %zu)",
          i, function_descriptor->bytecode_offset,
          flatbuffers_uint8_vec_len(bytecode_data));
    }
    if (function_descriptor->i32_register_count > IREE_I32_REGISTER_COUNT ||
        function_descriptor->ref_register_count > IREE_REF_REGISTER_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "functions[%zu] descriptor register count out of range", i);
    }

    // TODO(benvanik): run bytecode verifier on contents.
  }

  return iree_ok_status();
}

static void iree_vm_bytecode_module_destroy(void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;

  iree_allocator_free(module->flatbuffer_allocator,
                      (void*)module->flatbuffer_data.data);
  module->flatbuffer_data = iree_make_const_byte_span(NULL, 0);
  module->flatbuffer_allocator = iree_allocator_null();

  iree_allocator_free(module->allocator, module);
}

static iree_string_view_t iree_vm_bytecode_module_name(void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  flatbuffers_string_t name = iree_vm_BytecodeModuleDef_name(module->def);
  return iree_make_string_view(name, flatbuffers_string_len(name));
}

static iree_vm_module_signature_t iree_vm_bytecode_module_signature(
    void* self) {
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  iree_vm_module_signature_t signature;
  memset(&signature, 0, sizeof(signature));
  signature.import_function_count = iree_vm_ImportFunctionDef_vec_len(
      iree_vm_BytecodeModuleDef_imported_functions(module->def));
  signature.export_function_count = iree_vm_ExportFunctionDef_vec_len(
      iree_vm_BytecodeModuleDef_exported_functions(module->def));
  signature.internal_function_count = iree_vm_InternalFunctionDef_vec_len(
      iree_vm_BytecodeModuleDef_internal_functions(module->def));
  return signature;
}

static iree_status_t iree_vm_bytecode_module_get_function(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_vm_function_t* out_function, iree_string_view_t* out_name,
    iree_vm_function_signature_t* out_signature) {
  if (out_function) {
    memset(out_function, 0, sizeof(*out_function));
  }
  if (out_name) {
    memset(out_name, 0, sizeof(*out_name));
  }
  if (out_signature) {
    memset(out_signature, 0, sizeof(*out_signature));
  }

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  flatbuffers_string_t name = NULL;
  iree_vm_FunctionSignatureDef_table_t signature = NULL;
  if (linkage == IREE_VM_FUNCTION_LINKAGE_IMPORT) {
    iree_vm_ImportFunctionDef_vec_t imported_functions =
        iree_vm_BytecodeModuleDef_imported_functions(module->def);
    if (ordinal >= iree_vm_ImportFunctionDef_vec_len(imported_functions)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "import ordinal out of range (0 < %zu < %zu)", ordinal,
          iree_vm_ImportFunctionDef_vec_len(imported_functions));
    }
    iree_vm_ImportFunctionDef_table_t import_def =
        iree_vm_ImportFunctionDef_vec_at(imported_functions, ordinal);
    name = iree_vm_ImportFunctionDef_full_name(import_def);
    signature = iree_vm_ImportFunctionDef_signature(import_def);
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = linkage;
      out_function->ordinal = (uint16_t)ordinal;
    }
  } else if (linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT) {
    iree_vm_ExportFunctionDef_vec_t exported_functions =
        iree_vm_BytecodeModuleDef_exported_functions(module->def);
    if (ordinal >= iree_vm_ExportFunctionDef_vec_len(exported_functions)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export ordinal out of range (0 < %zu < %zu)", ordinal,
          iree_vm_ExportFunctionDef_vec_len(exported_functions));
    }
    iree_vm_ExportFunctionDef_table_t export_def =
        iree_vm_ExportFunctionDef_vec_at(exported_functions, ordinal);
    name = iree_vm_ExportFunctionDef_local_name(export_def);
    signature = iree_vm_ExportFunctionDef_signature(export_def);
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
      out_function->ordinal =
          iree_vm_ExportFunctionDef_internal_ordinal(export_def);

      const iree_vm_FunctionDescriptor_t* function_descriptor =
          &module->function_descriptor_table[out_function->ordinal];
      out_function->i32_register_count =
          function_descriptor->i32_register_count;
      out_function->ref_register_count =
          function_descriptor->ref_register_count;
    }
  } else {
    iree_vm_InternalFunctionDef_vec_t internal_functions =
        iree_vm_BytecodeModuleDef_internal_functions(module->def);
    if (ordinal >= iree_vm_InternalFunctionDef_vec_len(internal_functions)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function ordinal out of range (0 < %zu < %zu)", ordinal,
          iree_vm_InternalFunctionDef_vec_len(internal_functions));
    }
    iree_vm_InternalFunctionDef_table_t function_def =
        iree_vm_InternalFunctionDef_vec_at(internal_functions, ordinal);
    name = iree_vm_InternalFunctionDef_local_name(function_def);
    signature = iree_vm_InternalFunctionDef_signature(function_def);
    if (out_function) {
      out_function->module = &module->interface;
      out_function->linkage = IREE_VM_FUNCTION_LINKAGE_INTERNAL;
      out_function->ordinal = (uint16_t)ordinal;

      const iree_vm_FunctionDescriptor_t* function_descriptor =
          &module->function_descriptor_table[ordinal];
      out_function->i32_register_count =
          function_descriptor->i32_register_count;
      out_function->ref_register_count =
          function_descriptor->ref_register_count;
    }
  }

  if (out_name && name) {
    out_name->data = name;
    out_name->size = flatbuffers_string_len(name);
  }
  if (out_signature && signature) {
    out_signature->argument_count = flatbuffers_int32_vec_len(
        iree_vm_FunctionSignatureDef_argument_types(signature));
    out_signature->result_count = flatbuffers_int32_vec_len(
        iree_vm_FunctionSignatureDef_result_types(signature));
  }

  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_get_function_reflection_attr(
    void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
    iree_host_size_t index, iree_string_view_t* key,
    iree_string_view_t* value) {
  if (linkage != IREE_VM_FUNCTION_LINKAGE_INTERNAL) {
    iree_vm_function_t internal_function;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_get_function(
        self, linkage, ordinal, &internal_function, NULL, NULL));
    ordinal = internal_function.ordinal;
  }

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  iree_vm_InternalFunctionDef_vec_t internal_functions =
      iree_vm_BytecodeModuleDef_internal_functions(module->def);

  if (ordinal >= iree_vm_InternalFunctionDef_vec_len(internal_functions)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function ordinal out of range (0 < %zu < %zu)", ordinal,
        iree_vm_InternalFunctionDef_vec_len(internal_functions));
  }

  iree_vm_InternalFunctionDef_table_t function_def =
      iree_vm_InternalFunctionDef_vec_at(internal_functions, ordinal);
  iree_vm_FunctionSignatureDef_table_t signature =
      iree_vm_InternalFunctionDef_signature(function_def);
  iree_vm_ReflectionAttrDef_vec_t reflection_attrs =
      iree_vm_FunctionSignatureDef_reflection_attrs(signature);
  if (index >= iree_vm_ReflectionAttrDef_vec_len(reflection_attrs)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "reflection attribute at index %zu not found",
                            index);
  }
  iree_vm_ReflectionAttrDef_table_t attr =
      iree_vm_ReflectionAttrDef_vec_at(reflection_attrs, index);
  flatbuffers_string_t attr_key = iree_vm_ReflectionAttrDef_key(attr);
  flatbuffers_string_t attr_value = iree_vm_ReflectionAttrDef_value(attr);
  if (!flatbuffers_string_len(attr_key) ||
      !flatbuffers_string_len(attr_value)) {
    // Because reflection metadata should not impose any overhead for the
    // non reflection case, we do not eagerly validate it on load -- instead
    // verify it structurally as needed.
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "reflection attribute missing fields");
  }

  key->data = attr_key;
  key->size = flatbuffers_string_len(attr_key);
  value->data = attr_value;
  value->size = flatbuffers_string_len(attr_value);

  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_lookup_function(
    void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
    iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(out_function);
  memset(out_function, 0, sizeof(iree_vm_function_t));

  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function name required for query");
  }

  // NOTE: we could organize imports/exports alphabetically so we could bsearch.
  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  if (linkage == IREE_VM_FUNCTION_LINKAGE_IMPORT) {
    iree_vm_ImportFunctionDef_vec_t imported_functions =
        iree_vm_BytecodeModuleDef_imported_functions(module->def);
    for (size_t ordinal = 0;
         ordinal < iree_vm_ImportFunctionDef_vec_len(imported_functions);
         ++ordinal) {
      iree_vm_ImportFunctionDef_table_t import_def =
          iree_vm_ImportFunctionDef_vec_at(imported_functions, ordinal);
      if (iree_vm_flatbuffer_strcmp(
              iree_vm_ImportFunctionDef_full_name(import_def), name) == 0) {
        return iree_vm_bytecode_module_get_function(self, linkage, ordinal,
                                                    out_function, NULL, NULL);
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "import with the given name not found");
  } else if (linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT) {
    iree_vm_ExportFunctionDef_vec_t exported_functions =
        iree_vm_BytecodeModuleDef_exported_functions(module->def);
    for (size_t ordinal = 0;
         ordinal < iree_vm_InternalFunctionDef_vec_len(exported_functions);
         ++ordinal) {
      iree_vm_ExportFunctionDef_table_t export_def =
          iree_vm_ExportFunctionDef_vec_at(exported_functions, ordinal);
      if (iree_vm_flatbuffer_strcmp(
              iree_vm_ExportFunctionDef_local_name(export_def), name) == 0) {
        return iree_vm_bytecode_module_get_function(
            self, IREE_VM_FUNCTION_LINKAGE_INTERNAL,
            iree_vm_ExportFunctionDef_internal_ordinal(export_def),
            out_function, NULL, NULL);
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "export with the given name not found");
  } else {
    iree_vm_InternalFunctionDef_vec_t internal_functions =
        iree_vm_BytecodeModuleDef_internal_functions(module->def);
    for (size_t ordinal = 0;
         ordinal < iree_vm_InternalFunctionDef_vec_len(internal_functions);
         ++ordinal) {
      iree_vm_InternalFunctionDef_table_t function_def =
          iree_vm_InternalFunctionDef_vec_at(internal_functions, ordinal);
      if (iree_vm_flatbuffer_strcmp(
              iree_vm_InternalFunctionDef_local_name(function_def), name) ==
          0) {
        return iree_vm_bytecode_module_get_function(
            self, IREE_VM_FUNCTION_LINKAGE_INTERNAL, ordinal, out_function,
            NULL, NULL);
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "function with the given name not found");
  }
}

// Lays out the nested tables within a |state| structure.
// Returns the total size of the structure and all tables with padding applied.
// |state| may be null if only the structure size is required for allocation.
static iree_host_size_t iree_vm_bytecode_module_layout_state(
    iree_vm_BytecodeModuleDef_table_t module_def,
    iree_vm_bytecode_module_state_t* state) {
  iree_vm_ModuleStateDef_table_t module_state =
      iree_vm_BytecodeModuleDef_module_state(module_def);
  iree_host_size_t rwdata_storage_capacity = 0;
  iree_host_size_t global_ref_count = 0;
  if (module_state) {
    rwdata_storage_capacity =
        iree_vm_ModuleStateDef_global_bytes_capacity(module_state);
    global_ref_count = iree_vm_ModuleStateDef_global_ref_count(module_state);
  }
  iree_host_size_t rodata_ref_count = iree_vm_RodataSegmentDef_vec_len(
      iree_vm_BytecodeModuleDef_rodata_segments(module_def));
  iree_host_size_t import_function_count = iree_vm_ImportFunctionDef_vec_len(
      iree_vm_BytecodeModuleDef_imported_functions(module_def));

  uint8_t* base_ptr = (uint8_t*)state;
  iree_host_size_t offset =
      iree_align(sizeof(iree_vm_bytecode_module_state_t), 16);

  if (state) {
    state->rwdata_storage =
        iree_make_byte_span(base_ptr + offset, rwdata_storage_capacity);
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
  IREE_ASSERT_ARGUMENT(out_module_state);
  *out_module_state = NULL;

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  iree_vm_BytecodeModuleDef_table_t module_def = module->def;

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
  iree_vm_RodataSegmentDef_vec_t rodata_segments =
      iree_vm_BytecodeModuleDef_rodata_segments(module_def);
  for (int i = 0; i < state->rodata_ref_count; ++i) {
    iree_vm_RodataSegmentDef_table_t segment =
        iree_vm_RodataSegmentDef_vec_at(rodata_segments, i);
    iree_vm_ro_byte_buffer_t* ref = &state->rodata_ref_table[i];
    iree_atomic_store(&ref->ref_object.counter, 1);
    ref->data.data = iree_vm_RodataSegmentDef_data(segment);
    ref->data.data_length =
        flatbuffers_uint8_vec_len(iree_vm_RodataSegmentDef_data(segment));
  }

  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
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
    void* self, iree_vm_module_state_t* module_state, iree_host_size_t ordinal,
    iree_vm_function_t function) {
  IREE_ASSERT_ARGUMENT(module_state);
  iree_vm_bytecode_module_state_t* state =
      (iree_vm_bytecode_module_state_t*)module_state;
  if (ordinal >= state->import_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import ordinal out of range (0 < %zu < %zu)",
                            ordinal, state->import_count);
  }
  // TODO(benvanik): verify signature.
  state->import_table[ordinal] = function;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_begin_call(
    void* self, iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // NOTE: any work here adds directly to the invocation time. Avoid doing too
  // much work or touching too many unlikely-to-be-cached structures (such as
  // walking the FlatBuffer, which may cause page faults).

  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(iree_vm_execution_result_t));

  // Only internal functions store the information needed for execution. We
  // allow exports here as well to make things easier to call externally.
  iree_vm_function_t function = call->function;
  if (function.linkage != IREE_VM_FUNCTION_LINKAGE_INTERNAL) {
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_get_function(
        self, function.linkage, function.ordinal, &function, NULL, NULL));
  }

  iree_vm_bytecode_module_t* module = (iree_vm_bytecode_module_t*)self;
  if (function.ordinal >= module->function_descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function ordinal out of range (0 < %u < %zu)",
                            function.ordinal,
                            module->function_descriptor_count);
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
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_flatbuffer_verify(flatbuffer_data));

  iree_vm_BytecodeModuleDef_table_t module_def =
      iree_vm_BytecodeModuleDef_as_root(flatbuffer_data.data);
  if (!module_def) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "failed getting root from flatbuffer; expected identifier "
        "'" iree_vm_BytecodeModuleDef_file_identifier "' not found");
  }

  iree_vm_TypeDef_vec_t type_defs = iree_vm_BytecodeModuleDef_types(module_def);
  size_t type_table_size =
      iree_vm_TypeDef_vec_len(type_defs) * sizeof(iree_vm_type_def_t);

  iree_vm_bytecode_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(iree_vm_bytecode_module_t) + type_table_size,
      (void**)&module));
  module->allocator = allocator;

  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  module->function_descriptor_count =
      iree_vm_FunctionDescriptor_vec_len(function_descriptors);
  module->function_descriptor_table = function_descriptors;

  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  module->bytecode_data = iree_make_const_byte_span(
      bytecode_data, flatbuffers_uint8_vec_len(bytecode_data));

  module->flatbuffer_data = flatbuffer_data;
  module->flatbuffer_allocator = flatbuffer_allocator;
  module->def = module_def;

  module->type_count = iree_vm_TypeDef_vec_len(type_defs);
  module->type_table = (iree_vm_type_def_t*)((uint8_t*)module +
                                             sizeof(iree_vm_bytecode_module_t));
  iree_status_t resolve_status =
      iree_vm_bytecode_module_resolve_types(type_defs, module->type_table);
  if (!iree_status_is_ok(resolve_status)) {
    iree_allocator_free(allocator, module);
    return resolve_status;
  }

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
  return iree_ok_status();
}
