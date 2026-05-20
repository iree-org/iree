// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler/module.h"

#include <string.h>

static iree_status_t iree_vm_bytecode_assembler_reserve_labels(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->label_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->label_capacity ? state->label_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->labels[0]),
      (void**)&state->labels));
  state->label_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_fixups(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->fixup_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->fixup_capacity ? state->fixup_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->fixups[0]),
      (void**)&state->fixups));
  state->fixup_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_globals(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->global_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->global_capacity ? state->global_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->globals[0]),
      (void**)&state->globals));
  state->global_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_global_fixups(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->global_fixup_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      state->global_fixup_capacity ? state->global_fixup_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->global_fixups[0]),
      (void**)&state->global_fixups));
  state->global_fixup_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_types(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->type_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->type_capacity ? state->type_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->types[0]),
      (void**)&state->types));
  state->type_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_imports(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->import_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->import_capacity ? state->import_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->imports[0]),
      (void**)&state->imports));
  state->import_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_rodata_segments(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->rodata_segment_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      state->rodata_segment_capacity ? state->rodata_segment_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->rodata_segments[0]),
      (void**)&state->rodata_segments));
  state->rodata_segment_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_exports(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->export_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->export_capacity ? state->export_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->exports[0]),
      (void**)&state->exports));
  state->export_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_reserve_functions(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->function_capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      state->function_capacity ? state->function_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->functions[0]),
      (void**)&state->functions));
  state->function_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_assembler_reserve_function_fixups(
    iree_vm_bytecode_assembler_module_t* state,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->function_fixup_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      state->function_fixup_capacity ? state->function_fixup_capacity : 8;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      state->host_allocator, new_capacity, sizeof(state->function_fixups[0]),
      (void**)&state->function_fixups));
  state->function_fixup_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_label(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t name,
    uint32_t pc) {
  for (iree_host_size_t i = 0; i < state->label_count; ++i) {
    if (iree_string_view_equal(state->labels[i].name, name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate block label ^%.*s", (int)name.size,
                              name.data);
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_reserve_labels(state, state->label_count + 1));
  state->labels[state->label_count++] =
      (iree_vm_bytecode_assembler_label_t){.name = name, .pc = pc};
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_fixup(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t label_name,
    uint32_t bytecode_offset) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_reserve_fixups(state, state->fixup_count + 1));
  state->fixups[state->fixup_count++] = (iree_vm_bytecode_assembler_fixup_t){
      .label_name = label_name, .bytecode_offset = bytecode_offset};
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_global(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t name,
    uint32_t storage_size) {
  for (iree_host_size_t i = 0; i < state->global_count; ++i) {
    if (iree_string_view_equal(state->globals[i].name, name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate global @%.*s", (int)name.size,
                              name.data);
    }
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_globals(
      state, state->global_count + 1));
  state->globals[state->global_count++] = (iree_vm_bytecode_assembler_global_t){
      .name = name,
      .storage_size = storage_size,
      .ordinal = UINT32_MAX,
  };
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_global_fixup(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t name,
    uint32_t bytecode_offset, uint32_t storage_size) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_global_fixups(
      state, state->global_fixup_count + 1));
  state->global_fixups[state->global_fixup_count++] =
      (iree_vm_bytecode_assembler_global_fixup_t){
          .name = name,
          .bytecode_offset = bytecode_offset,
          .storage_size = storage_size,
      };
  return iree_ok_status();
}

iree_host_size_t iree_vm_bytecode_assembler_find_type_ordinal(
    const iree_vm_bytecode_assembler_module_t* state,
    iree_string_view_t full_name) {
  for (iree_host_size_t i = 0; i < state->type_count; ++i) {
    if (iree_string_view_equal(state->types[i].full_name, full_name)) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t iree_vm_bytecode_assembler_append_type(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t full_name,
    iree_host_size_t* out_ordinal) {
  if (iree_vm_bytecode_assembler_find_type_ordinal(state, full_name) !=
      IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "duplicate type %.*s",
                            (int)full_name.size, full_name.data);
  }
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_reserve_types(state, state->type_count + 1));
  const iree_host_size_t ordinal = state->type_count;
  state->types[state->type_count++] =
      (iree_vm_bytecode_assembler_type_t){.full_name = full_name};
  if (out_ordinal) *out_ordinal = ordinal;
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_lookup_or_append_type(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t full_name,
    iree_host_size_t* out_ordinal) {
  iree_host_size_t ordinal =
      iree_vm_bytecode_assembler_find_type_ordinal(state, full_name);
  if (ordinal == IREE_HOST_SIZE_MAX) {
    return iree_vm_bytecode_assembler_append_type(state, full_name,
                                                  out_ordinal);
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

iree_host_size_t iree_vm_bytecode_assembler_find_import_ordinal(
    const iree_vm_bytecode_assembler_module_t* state,
    iree_string_view_t full_name) {
  for (iree_host_size_t i = 0; i < state->import_count; ++i) {
    if (iree_string_view_equal(state->imports[i].full_name, full_name)) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t iree_vm_bytecode_assembler_append_import(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t full_name,
    iree_string_view_t calling_convention, bool optional) {
  if (iree_vm_bytecode_assembler_find_import_ordinal(state, full_name) !=
      IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate import @%.*s", (int)full_name.size,
                            full_name.data);
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_imports(
      state, state->import_count + 1));
  state->imports[state->import_count++] = (iree_vm_bytecode_assembler_import_t){
      .full_name = full_name,
      .calling_convention = calling_convention,
      .optional = optional,
  };
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_rodata_segment(
    iree_vm_bytecode_assembler_module_t* state, iree_host_size_t ordinal,
    iree_const_byte_span_t data) {
  if (ordinal != state->rodata_segment_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "rodata segments must be declared in ordinal order");
  }
  if (data.data_length > 0 && !data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rodata segment has no bytes");
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_rodata_segments(
      state, state->rodata_segment_count + 1));
  uint8_t* segment_data = NULL;
  if (data.data_length > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        state->host_allocator, data.data_length, (void**)&segment_data));
    memcpy(segment_data, data.data, data.data_length);
  }
  state->rodata_segments[state->rodata_segment_count++] =
      (iree_vm_bytecode_assembler_rodata_t){
          .data = segment_data,
          .data_length = data.data_length,
      };
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_clone_string(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t value,
    iree_string_view_t* out_value) {
  char* data = NULL;
  if (value.size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(state->host_allocator,
                                               value.size, (void**)&data));
    memcpy(data, value.data, value.size);
  }
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

iree_host_size_t iree_vm_bytecode_assembler_find_function_ordinal(
    const iree_vm_bytecode_assembler_module_t* state, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < state->function_count; ++i) {
    if (iree_string_view_equal(state->functions[i].name, name)) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

iree_status_t iree_vm_bytecode_assembler_append_function_fixup(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t name,
    uint32_t bytecode_offset) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_function_fixups(
      state, state->function_fixup_count + 1));
  state->function_fixups[state->function_fixup_count++] =
      (iree_vm_bytecode_assembler_function_fixup_t){
          .name = name,
          .bytecode_offset = bytecode_offset,
      };
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_append_export(
    iree_vm_bytecode_assembler_module_t* state, iree_string_view_t export_name,
    iree_string_view_t function_name) {
  for (iree_host_size_t i = 0; i < state->export_count; ++i) {
    if (iree_string_view_equal(state->exports[i].name, export_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate export @%.*s", (int)export_name.size,
                              export_name.data);
    }
  }
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_assembler_reserve_exports(
      state, state->export_count + 1));
  state->exports[state->export_count++] = (iree_vm_bytecode_assembler_export_t){
      .name = export_name,
      .function_name = function_name,
  };
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_resolve_function_fixups(
    iree_vm_bytecode_assembler_module_t* state) {
  for (iree_host_size_t i = 0; i < state->function_fixup_count; ++i) {
    const iree_vm_bytecode_assembler_function_fixup_t* fixup =
        &state->function_fixups[i];
    const iree_host_size_t ordinal =
        iree_vm_bytecode_assembler_find_function_ordinal(state, fixup->name);
    if (ordinal == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown function @%.*s", (int)fixup->name.size,
                              fixup->name.data);
    }
    if (ordinal > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function ordinal is too large");
    }
    if (fixup->bytecode_offset + 4 >
        iree_string_builder_size(&state->bytecode_builder)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function fixup is out of range");
    }
    uint8_t* target =
        (uint8_t*)state->bytecode_builder.buffer + fixup->bytecode_offset;
    target[0] = (uint8_t)(ordinal & 0xFFu);
    target[1] = (uint8_t)((ordinal >> 8) & 0xFFu);
    target[2] = (uint8_t)((ordinal >> 16) & 0xFFu);
    target[3] = (uint8_t)((ordinal >> 24) & 0xFFu);
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_assign_global_ordinals(
    iree_vm_bytecode_assembler_module_t* state) {
  uint32_t next_global_ref_ordinal = 0;
  for (iree_host_size_t i = 0; i < state->global_count; ++i) {
    iree_vm_bytecode_assembler_global_t* global = &state->globals[i];
    if (global->storage_size != 0) continue;
    global->ordinal = next_global_ref_ordinal++;
  }
  if (next_global_ref_ordinal > state->global_ref_count) {
    state->global_ref_count = next_global_ref_ordinal;
  }

  uint32_t next_global_byte_ordinal = 0;
  for (uint32_t storage_size = 1; storage_size <= 8; ++storage_size) {
    bool has_sized_global = false;
    for (iree_host_size_t i = 0; i < state->global_count; ++i) {
      if (state->globals[i].storage_size == storage_size) {
        has_sized_global = true;
        break;
      }
    }
    if (!has_sized_global) continue;

    next_global_byte_ordinal =
        (next_global_byte_ordinal + storage_size - 1) & ~(storage_size - 1);
    for (iree_host_size_t i = 0; i < state->global_count; ++i) {
      iree_vm_bytecode_assembler_global_t* global = &state->globals[i];
      if (global->storage_size != storage_size) continue;
      global->ordinal = next_global_byte_ordinal;
      if (next_global_byte_ordinal > UINT32_MAX - storage_size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "global byte storage is too large");
      }
      next_global_byte_ordinal += storage_size;
    }
  }
  if (next_global_byte_ordinal > state->global_byte_capacity) {
    state->global_byte_capacity = next_global_byte_ordinal;
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_resolve_global_fixups(
    iree_vm_bytecode_assembler_module_t* state) {
  for (iree_host_size_t i = 0; i < state->global_fixup_count; ++i) {
    const iree_vm_bytecode_assembler_global_fixup_t* fixup =
        &state->global_fixups[i];
    const iree_vm_bytecode_assembler_global_t* global = NULL;
    for (iree_host_size_t j = 0; j < state->global_count; ++j) {
      if (iree_string_view_equal(state->globals[j].name, fixup->name)) {
        global = &state->globals[j];
        break;
      }
    }
    if (!global) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown global @%.*s", (int)fixup->name.size,
                              fixup->name.data);
    }
    if (global->storage_size != fixup->storage_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "global @%.*s storage class mismatch",
                              (int)fixup->name.size, fixup->name.data);
    }
    if (fixup->bytecode_offset + 4 >
        iree_string_builder_size(&state->bytecode_builder)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "global fixup is out of range");
    }
    uint8_t* target =
        (uint8_t*)state->bytecode_builder.buffer + fixup->bytecode_offset;
    target[0] = (uint8_t)(global->ordinal & 0xFFu);
    target[1] = (uint8_t)((global->ordinal >> 8) & 0xFFu);
    target[2] = (uint8_t)((global->ordinal >> 16) & 0xFFu);
    target[3] = (uint8_t)((global->ordinal >> 24) & 0xFFu);
  }
  return iree_ok_status();
}

void iree_vm_bytecode_assembler_module_initialize(
    iree_allocator_t host_allocator,
    iree_vm_bytecode_assembler_module_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->host_allocator = host_allocator;
  iree_string_builder_initialize(host_allocator,
                                 &out_state->argument_cconv_builder);
  iree_string_builder_initialize(host_allocator,
                                 &out_state->result_cconv_builder);
  iree_string_builder_initialize(host_allocator, &out_state->bytecode_builder);
  iree_string_builder_initialize(host_allocator, &out_state->cconv_builder);
}

void iree_vm_bytecode_assembler_module_deinitialize(
    iree_vm_bytecode_assembler_module_t* state) {
  iree_allocator_t host_allocator = state->host_allocator;
  iree_string_builder_deinitialize(&state->cconv_builder);
  iree_string_builder_deinitialize(&state->bytecode_builder);
  iree_string_builder_deinitialize(&state->result_cconv_builder);
  iree_string_builder_deinitialize(&state->argument_cconv_builder);
  for (iree_host_size_t i = 0; i < state->function_count; ++i) {
    iree_allocator_free(host_allocator, (void*)state->functions[i].cconv.data);
  }
  for (iree_host_size_t i = 0; i < state->rodata_segment_count; ++i) {
    iree_allocator_free(host_allocator, state->rodata_segments[i].data);
  }
  iree_allocator_free(host_allocator, state->function_fixups);
  iree_allocator_free(host_allocator, state->functions);
  iree_allocator_free(host_allocator, state->exports);
  iree_allocator_free(host_allocator, state->rodata_segments);
  iree_allocator_free(host_allocator, state->imports);
  iree_allocator_free(host_allocator, state->types);
  iree_allocator_free(host_allocator, state->global_fixups);
  iree_allocator_free(host_allocator, state->globals);
  iree_allocator_free(host_allocator, state->fixups);
  iree_allocator_free(host_allocator, state->labels);
}
