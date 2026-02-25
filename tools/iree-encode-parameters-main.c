// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/io/stream.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/api.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(bool, list_targets, false,
          "Lists the targets an encoding module can produce parameters for and "
          "exit.");

IREE_FLAG(bool, list_parameters, false,
          "Lists the parameters that will be encoded and exit.");

IREE_FLAG(string, target, "",
          "Target to use for encoding. If not specified, uses auto-detection.");

IREE_FLAG(bool, quiet, false,
          "Suppress output except for errors. Exit code indicates success.");

IREE_FLAG_LIST(string, output,
               "Specifies an output parameter file per scope.\n"
               "Format: `scope=path.irpa` or `path.irpa` for default scope.\n"
               "Example: `--output=encoded=output.irpa`");

//===----------------------------------------------------------------------===//
// Encoder target discovery
//===----------------------------------------------------------------------===//

// Encoder function set for a single target.
typedef struct iree_encode_target_t {
  iree_string_view_t target;
  iree_vm_function_t indices_fn;
  iree_vm_function_t steps_fn;
  iree_vm_function_t encode_fn;
} iree_encode_target_t;

// Storage for discovered encoder targets.
typedef struct iree_encode_target_set_t {
  iree_vm_function_t detect_target_fn;
  iree_host_size_t target_count;
  iree_host_size_t target_capacity;
  iree_encode_target_t* targets;
  iree_allocator_t allocator;
} iree_encode_target_set_t;

static void iree_encode_target_set_initialize(
    iree_allocator_t allocator, iree_encode_target_set_t* out_target_set) {
  memset(out_target_set, 0, sizeof(*out_target_set));
  out_target_set->allocator = allocator;
}

static void iree_encode_target_set_deinitialize(
    iree_encode_target_set_t* target_set) {
  if (target_set->targets) {
    iree_allocator_free(target_set->allocator, target_set->targets);
  }
  memset(target_set, 0, sizeof(*target_set));
}

static iree_status_t iree_encode_target_set_add(
    iree_encode_target_set_t* target_set, iree_string_view_t target_name,
    iree_encode_target_t** out_target) {
  // Check if target already exists.
  for (iree_host_size_t i = 0; i < target_set->target_count; ++i) {
    if (iree_string_view_equal(target_set->targets[i].target, target_name)) {
      *out_target = &target_set->targets[i];
      return iree_ok_status();
    }
  }
  // Grow if needed.
  if (target_set->target_count >= target_set->target_capacity) {
    iree_host_size_t new_capacity =
        target_set->target_capacity ? target_set->target_capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        target_set->allocator, new_capacity * sizeof(iree_encode_target_t),
        (void**)&target_set->targets));
    target_set->target_capacity = new_capacity;
  }
  // Add new target.
  iree_encode_target_t* target = &target_set->targets[target_set->target_count];
  memset(target, 0, sizeof(*target));
  target->target = target_name;
  ++target_set->target_count;
  *out_target = target;
  return iree_ok_status();
}

// Looks up a reflection attribute value by key.
static iree_string_view_t iree_encode_lookup_reflection_attr(
    iree_vm_function_t* function, iree_string_view_t key) {
  return iree_vm_function_lookup_attr_by_name(function, key);
}

// Discovers encoder functions from the module by scanning exported function
// attributes.
static iree_status_t iree_encode_discover_functions(
    iree_vm_module_t* module, iree_encode_target_set_t* target_set) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_module_signature_t signature = iree_vm_module_signature(module);

  for (iree_host_size_t i = 0; i < signature.export_function_count; ++i) {
    iree_vm_function_t function;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_module_lookup_function_by_ordinal(
                module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));

    // Check for iree.encode.function attribute.
    iree_string_view_t encode_function = iree_encode_lookup_reflection_attr(
        &function, IREE_SV("iree.encode.function"));
    if (iree_string_view_is_empty(encode_function)) continue;

    if (iree_string_view_equal(encode_function, IREE_SV("detect_target"))) {
      target_set->detect_target_fn = function;
    } else {
      // Get target name for indices/steps/encode functions.
      iree_string_view_t target_name = iree_encode_lookup_reflection_attr(
          &function, IREE_SV("iree.encode.target"));
      if (iree_string_view_is_empty(target_name)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "encoder function missing iree.encode.target");
      }

      iree_encode_target_t* target = NULL;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_encode_target_set_add(target_set, target_name, &target));

      if (iree_string_view_equal(encode_function, IREE_SV("indices"))) {
        target->indices_fn = function;
      } else if (iree_string_view_equal(encode_function, IREE_SV("steps"))) {
        target->steps_fn = function;
      } else if (iree_string_view_equal(encode_function, IREE_SV("encode"))) {
        target->encode_fn = function;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Output scope/archive types
//===----------------------------------------------------------------------===//

typedef struct iree_output_scope_t {
  iree_string_view_t scope;
  iree_string_view_t path;
} iree_output_scope_t;

typedef struct iree_output_scope_list_t {
  iree_host_size_t count;
  iree_output_scope_t* entries;
  iree_allocator_t allocator;
} iree_output_scope_list_t;

static void iree_output_scope_list_initialize(iree_allocator_t allocator,
                                              iree_output_scope_list_t* list) {
  memset(list, 0, sizeof(*list));
  list->allocator = allocator;
}

static void iree_output_scope_list_deinitialize(
    iree_output_scope_list_t* list) {
  if (list->entries) {
    iree_allocator_free(list->allocator, list->entries);
  }
  memset(list, 0, sizeof(*list));
}

// Archive context for a single output scope.
typedef struct iree_output_archive_t {
  iree_string_view_t scope;
  iree_string_view_t path;
  iree_io_parameter_archive_builder_t builder;
  iree_io_file_handle_t* file_handle;
  iree_io_parameter_index_t* index;
  iree_io_parameter_provider_t* provider;
} iree_output_archive_t;

static void iree_output_archive_deinitialize(iree_output_archive_t* archive) {
  iree_io_parameter_provider_release(archive->provider);
  iree_io_parameter_index_release(archive->index);
  iree_io_file_handle_release(archive->file_handle);
  iree_io_parameter_archive_builder_deinitialize(&archive->builder);
}

//===----------------------------------------------------------------------===//
// Load modules and discover encoder functions
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_load_and_discover(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_tooling_module_list_t* out_module_list,
    iree_vm_module_t** out_encoder_module,
    iree_encode_target_set_t* out_target_set) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tooling_module_list_initialize(out_module_list);
  iree_encode_target_set_initialize(host_allocator, out_target_set);

  // Load modules from flags.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_load_modules_from_flags(instance, host_allocator,
                                               out_module_list));

  if (out_module_list->count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no modules specified; use --module=path.vmfb");
  }

  // Encoder module is the last module (by convention).
  *out_encoder_module = out_module_list->values[out_module_list->count - 1];

  // Discover encoder functions.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_encode_discover_functions(*out_encoder_module, out_target_set));

  if (out_target_set->target_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no encoder functions found in module; ensure the module was produced "
        "by iree-compile with --iree-parameter-encoder-output-file");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Select target
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_select_target(
    iree_encode_target_set_t* target_set,
    iree_encode_target_t** out_selected_target) {
  iree_string_view_t target_flag = iree_make_cstring_view(FLAG_target);

  if (iree_string_view_is_empty(target_flag)) {
    // Use first target.
    *out_selected_target = &target_set->targets[0];
    return iree_ok_status();
  }

  // Find matching target.
  for (iree_host_size_t i = 0; i < target_set->target_count; ++i) {
    if (iree_string_view_equal(target_set->targets[i].target, target_flag)) {
      *out_selected_target = &target_set->targets[i];
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "target '%s' not found in encoder module; "
                          "use --list-targets to see available targets",
                          FLAG_target);
}

static iree_status_t iree_encode_validate_target(iree_encode_target_t* target) {
  if (!target->indices_fn.module) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "indices function not found for target '%.*s'; "
                            "encoder module may be incomplete",
                            (int)target->target.size, target->target.data);
  }
  if (!target->encode_fn.module) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "encode function not found for target '%.*s'; "
                            "encoder module may be incomplete",
                            (int)target->target.size, target->target.data);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// --list_targets implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_print_targets(
    iree_vm_module_t* encoder_module, iree_encode_target_set_t* target_set) {
  iree_string_view_t module_name = iree_vm_module_name(encoder_module);
  fprintf(stdout, "Encoder module: %.*s\n", (int)module_name.size,
          module_name.data);
  fprintf(stdout, "Available targets:\n");

  for (iree_host_size_t i = 0; i < target_set->target_count; ++i) {
    iree_encode_target_t* target = &target_set->targets[i];
    fprintf(stdout, "  %.*s\n", (int)target->target.size, target->target.data);

    iree_string_view_t scopes = iree_encode_lookup_reflection_attr(
        &target->indices_fn, IREE_SV("iree.encode.scopes"));
    if (!iree_string_view_is_empty(scopes)) {
      fprintf(stdout, "    scopes: %.*s\n", (int)scopes.size, scopes.data);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Call indices function
//===----------------------------------------------------------------------===//

// Creates a temporary context and calls the indices function.
// The indices function returns constant data and doesn't need parameters.
// TODO(benvanik): Consider calling without full context if function has no
// imports.
static iree_status_t iree_encode_call_indices(
    iree_vm_instance_t* instance, iree_tooling_module_list_t* module_list,
    iree_encode_target_t* target, iree_allocator_t host_allocator,
    iree_vm_list_t** out_indices_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_create_context_from_flags(
              instance, module_list->count, module_list->values,
              /*default_device_uri=*/iree_string_view_empty(), host_allocator,
              &context, &device, /*out_device_allocator=*/NULL));

  // Invoke indices function.
  iree_vm_list_t* outputs = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, host_allocator, &outputs);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        context, target->indices_fn, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, /*inputs=*/NULL, outputs, host_allocator);
  }

  // Extract result list.
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t list_ref = iree_vm_ref_null();
    status = iree_vm_list_get_ref_assign(outputs, 0, &list_ref);
    if (iree_status_is_ok(status)) {
      *out_indices_list = iree_vm_list_deref(list_ref);
      if (*out_indices_list) {
        iree_vm_list_retain(*out_indices_list);
      }
    }
  }

  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// --list_parameters implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_print_parameters(
    iree_vm_list_t* indices_list) {
  iree_host_size_t scope_count = iree_vm_list_size(indices_list);

  for (iree_host_size_t scope_i = 0; scope_i < scope_count; ++scope_i) {
    iree_vm_ref_t scope_struct_ref = iree_vm_ref_null();
    if (!iree_status_is_ok(iree_vm_list_get_ref_assign(indices_list, scope_i,
                                                       &scope_struct_ref))) {
      continue;
    }
    iree_vm_list_t* scope_struct = iree_vm_list_deref(scope_struct_ref);
    if (!scope_struct || iree_vm_list_size(scope_struct) < 2) continue;

    // Get scope name.
    iree_vm_ref_t scope_name_ref = iree_vm_ref_null();
    iree_vm_list_get_ref_assign(scope_struct, 0, &scope_name_ref);
    iree_vm_buffer_t* scope_name_buffer = iree_vm_buffer_deref(scope_name_ref);
    iree_string_view_t scope_name =
        scope_name_buffer ? iree_vm_buffer_as_string(scope_name_buffer)
                          : IREE_SV("<default>");

    fprintf(stdout, "Scope: \"%.*s\"\n", (int)scope_name.size, scope_name.data);

    // Get entries list.
    iree_vm_ref_t entries_ref = iree_vm_ref_null();
    iree_vm_list_get_ref_assign(scope_struct, 1, &entries_ref);
    iree_vm_list_t* entries = iree_vm_list_deref(entries_ref);
    if (!entries) continue;

    // Print each entry.
    iree_host_size_t entry_count = iree_vm_list_size(entries);
    for (iree_host_size_t entry_i = 0; entry_i < entry_count; ++entry_i) {
      iree_vm_ref_t entry_ref = iree_vm_ref_null();
      if (!iree_status_is_ok(
              iree_vm_list_get_ref_assign(entries, entry_i, &entry_ref))) {
        continue;
      }
      iree_vm_list_t* entry = iree_vm_list_deref(entry_ref);
      if (!entry || iree_vm_list_size(entry) < 5) continue;

      iree_vm_value_t type_value, length_value;
      iree_vm_list_get_value(entry, 0, &type_value);
      iree_vm_list_get_value(entry, 3, &length_value);

      iree_vm_ref_t key_ref = iree_vm_ref_null();
      iree_vm_list_get_ref_assign(entry, 1, &key_ref);
      iree_vm_buffer_t* key_buffer = iree_vm_buffer_deref(key_ref);
      iree_string_view_t key = key_buffer ? iree_vm_buffer_as_string(key_buffer)
                                          : IREE_SV("<unknown>");

      if (type_value.i64 == 0) {
        // SPLAT entry.
        iree_vm_value_t pattern_value, pattern_length_value;
        iree_vm_list_get_value(entry, 4, &pattern_value);
        iree_vm_list_get_value(entry, 5, &pattern_length_value);
        fprintf(stdout,
                "  %.*s: SPLAT, %" PRIu64 " bytes, pattern=0x%0*" PRIx64 "\n",
                (int)key.size, key.data, (uint64_t)length_value.i64,
                (int)pattern_length_value.i64 * 2, (uint64_t)pattern_value.i64);
      } else {
        // DATA entry.
        iree_vm_value_t alignment_value;
        iree_vm_list_get_value(entry, 4, &alignment_value);
        fprintf(stdout,
                "  %.*s: DATA, %" PRIu64 " bytes, alignment %" PRIu64 "\n",
                (int)key.size, key.data, (uint64_t)length_value.i64,
                (uint64_t)alignment_value.i64);
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Parse output flags
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_parse_output_flags(
    iree_output_scope_list_t* list) {
  iree_host_size_t count = FLAG_output_list().count;
  if (count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      list->allocator, count * sizeof(iree_output_scope_t),
      (void**)&list->entries));
  list->count = count;

  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t flag = FLAG_output_list().values[i];
    iree_string_view_t scope, path;
    if (iree_string_view_split(flag, '=', &scope, &path) == -1) {
      // No scope provided - use empty scope.
      path = scope;
      scope = iree_string_view_empty();
    }
    list->entries[i].scope = scope;
    list->entries[i].path = path;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create output archives
//===----------------------------------------------------------------------===//

// Parses parameter indices and populates archive builders.
static iree_status_t iree_encode_parse_indices_into_archives(
    iree_vm_list_t* indices_list, iree_output_archive_t* archives,
    iree_host_size_t archive_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t scope_count = iree_vm_list_size(indices_list);
  for (iree_host_size_t scope_i = 0; scope_i < scope_count; ++scope_i) {
    // Get scope struct: [scope_name, entries_list].
    iree_vm_ref_t scope_struct_ref = iree_vm_ref_null();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_vm_list_get_ref_assign(indices_list, scope_i, &scope_struct_ref));
    iree_vm_list_t* scope_struct = iree_vm_list_deref(scope_struct_ref);
    if (!scope_struct || iree_vm_list_size(scope_struct) < 2) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid scope struct in indices");
    }

    // Get scope name.
    iree_vm_ref_t scope_name_ref = iree_vm_ref_null();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_ref_assign(scope_struct, 0, &scope_name_ref));
    iree_vm_buffer_t* scope_name_buffer = iree_vm_buffer_deref(scope_name_ref);
    iree_string_view_t scope_name =
        scope_name_buffer ? iree_vm_buffer_as_string(scope_name_buffer)
                          : iree_string_view_empty();

    // Find matching archive.
    iree_output_archive_t* archive = NULL;
    for (iree_host_size_t j = 0; j < archive_count; ++j) {
      if (iree_string_view_equal(archives[j].scope, scope_name)) {
        archive = &archives[j];
        break;
      }
    }
    if (!archive) continue;  // Scope not in output list.

    // Get entries list.
    iree_vm_ref_t entries_ref = iree_vm_ref_null();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_ref_assign(scope_struct, 1, &entries_ref));
    iree_vm_list_t* entries = iree_vm_list_deref(entries_ref);
    if (!entries) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid entries in scope struct");
    }

    // Process each parameter entry.
    iree_host_size_t entry_count = iree_vm_list_size(entries);
    for (iree_host_size_t entry_i = 0; entry_i < entry_count; ++entry_i) {
      iree_vm_ref_t entry_ref = iree_vm_ref_null();
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_vm_list_get_ref_assign(entries, entry_i, &entry_ref));
      iree_vm_list_t* entry = iree_vm_list_deref(entry_ref);
      if (!entry || iree_vm_list_size(entry) < 5) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid entry in entries list");
      }

      // Parse entry fields: [type, key, metadata, length, ...].
      iree_vm_value_t type_value, length_value;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_vm_list_get_value(entry, 0, &type_value));

      iree_vm_ref_t key_ref = iree_vm_ref_null();
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_vm_list_get_ref_assign(entry, 1, &key_ref));
      iree_vm_buffer_t* key_buffer = iree_vm_buffer_deref(key_ref);
      if (!key_buffer) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameter entry missing key");
      }
      iree_string_view_t key = iree_vm_buffer_as_string(key_buffer);

      iree_vm_ref_t metadata_ref = iree_vm_ref_null();
      iree_vm_list_get_ref_assign(entry, 2, &metadata_ref);
      iree_vm_buffer_t* metadata_buffer = iree_vm_buffer_deref(metadata_ref);
      iree_const_byte_span_t metadata = iree_const_byte_span_empty();
      if (metadata_buffer) {
        metadata = iree_vm_buffer_const_contents(metadata_buffer);
      }

      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_vm_list_get_value(entry, 3, &length_value));
      uint64_t length = (uint64_t)length_value.i64;

      if (type_value.i64 == 0) {
        // SPLAT entry: [type, key, metadata, length, pattern, pattern_length].
        iree_vm_value_t pattern_value, pattern_length_value;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_vm_list_get_value(entry, 4, &pattern_value));
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_vm_list_get_value(entry, 5, &pattern_length_value));

        uint64_t pattern = (uint64_t)pattern_value.i64;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_io_parameter_archive_builder_add_splat_entry(
                    &archive->builder, key, metadata, &pattern,
                    (uint8_t)pattern_length_value.i64, length));
      } else {
        // DATA entry: [type, key, metadata, length, alignment].
        iree_vm_value_t alignment_value;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_vm_list_get_value(entry, 4, &alignment_value));

        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_io_parameter_archive_builder_add_data_entry(
                    &archive->builder, key, metadata,
                    (uint64_t)alignment_value.i64, length));
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Creates archive files and providers for each output scope.
static iree_status_t iree_encode_create_archives(
    iree_vm_list_t* indices_list, iree_output_scope_list_t* output_list,
    iree_allocator_t host_allocator, iree_output_archive_t** out_archives) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate archive array.
  iree_output_archive_t* archives = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            output_list->count * sizeof(iree_output_archive_t),
                            (void**)&archives));

  // Initialize archive builders.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < output_list->count; ++i) {
    memset(&archives[i], 0, sizeof(archives[i]));
    archives[i].scope = output_list->entries[i].scope;
    archives[i].path = output_list->entries[i].path;
    status = iree_io_parameter_archive_builder_initialize(host_allocator,
                                                          &archives[i].builder);
    if (!iree_status_is_ok(status)) break;
  }

  // Parse indices into archive builders.
  if (iree_status_is_ok(status)) {
    status = iree_encode_parse_indices_into_archives(indices_list, archives,
                                                     output_list->count);
  }

  // Create files and write headers.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < output_list->count; ++i) {
      iree_output_archive_t* archive = &archives[i];

      iree_io_physical_size_t archive_size =
          iree_io_parameter_archive_builder_total_size(&archive->builder);

      // Create null-terminated path.
      char* path_cstr = NULL;
      status = iree_allocator_malloc(host_allocator, archive->path.size + 1,
                                     (void**)&path_cstr);
      if (!iree_status_is_ok(status)) break;
      memcpy(path_cstr, archive->path.data, archive->path.size);
      path_cstr[archive->path.size] = '\0';

      // Create output file.
      status = iree_io_file_handle_create(
          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE,
          iree_make_cstring_view(path_cstr), archive_size, host_allocator,
          &archive->file_handle);
      iree_allocator_free(host_allocator, path_cstr);
      if (!iree_status_is_ok(status)) break;

      // Create stream and index.
      iree_io_stream_t* stream = NULL;
      status =
          iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE,
                              archive->file_handle, 0, host_allocator, &stream);
      if (!iree_status_is_ok(status)) break;

      status = iree_io_parameter_index_create(host_allocator, &archive->index);
      if (!iree_status_is_ok(status)) {
        iree_io_stream_release(stream);
        break;
      }

      // Write archive header.
      status = iree_io_parameter_archive_builder_write(
          &archive->builder, archive->file_handle, 0, stream, archive->index);
      iree_io_stream_release(stream);
      if (!iree_status_is_ok(status)) break;

      // Create provider backed by the archive.
      status = iree_io_parameter_index_provider_create(
          archive->scope, archive->index,
          IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
          host_allocator, &archive->provider);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < output_list->count; ++i) {
      iree_output_archive_deinitialize(&archives[i]);
    }
    iree_allocator_free(host_allocator, archives);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_archives = archives;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create encoding context with output providers
//===----------------------------------------------------------------------===//

// Creates the encoding context with output providers attached.
// TODO(benvanik): Allow adding providers to existing parameters module to avoid
// recreating context.
static iree_status_t iree_encode_create_encoding_context(
    iree_vm_instance_t* instance, iree_tooling_module_list_t* module_list,
    iree_output_archive_t* archives, iree_host_size_t archive_count,
    iree_allocator_t host_allocator, iree_vm_context_t** out_context,
    iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Collect output providers.
  iree_host_size_t provider_count = 0;
  for (iree_host_size_t i = 0; i < archive_count; ++i) {
    if (archives[i].provider) ++provider_count;
  }

  iree_io_parameter_provider_t** providers =
      (iree_io_parameter_provider_t**)iree_alloca(
          provider_count * sizeof(iree_io_parameter_provider_t*));
  for (iree_host_size_t i = 0, j = 0; i < archive_count; ++i) {
    if (archives[i].provider) {
      providers[j++] = archives[i].provider;
    }
  }

  // Create parameters module with output providers.
  iree_vm_module_t* params_module = NULL;
  iree_status_t status = iree_tooling_create_parameters_module_from_flags(
      instance, provider_count, providers, host_allocator, &params_module);

  // Pre-populate resolved_list with params module so resolver won't create
  // default.
  iree_tooling_module_list_t resolved_list;
  iree_tooling_module_list_initialize(&resolved_list);

  if (iree_status_is_ok(status)) {
    status = iree_tooling_module_list_push_back(&resolved_list, params_module);
  }

  // Resolve dependencies (adds HAL, etc.).
  if (iree_status_is_ok(status)) {
    status = iree_tooling_resolve_modules(
        instance, module_list->count, module_list->values,
        /*default_device_uri=*/iree_string_view_empty(), host_allocator,
        &resolved_list, out_device, /*out_device_allocator=*/NULL);
  }

  // Create context.
  if (iree_status_is_ok(status)) {
    status = iree_vm_context_create_with_modules(
        instance, IREE_VM_CONTEXT_FLAG_NONE, resolved_list.count,
        resolved_list.values, host_allocator, out_context);
  }

  iree_tooling_module_list_reset(&resolved_list);
  iree_vm_module_release(params_module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Call steps function
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_call_steps(iree_vm_context_t* context,
                                            iree_encode_target_t* target,
                                            iree_allocator_t host_allocator,
                                            iree_vm_list_t** out_steps_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_steps_list = NULL;
  if (!target->steps_fn.module) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();  // Steps function is optional.
  }

  iree_vm_list_t* outputs = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, host_allocator, &outputs);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        context, target->steps_fn, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, /*inputs=*/NULL, outputs, host_allocator);
  }

  if (iree_status_is_ok(status)) {
    iree_vm_ref_t list_ref = iree_vm_ref_null();
    status = iree_vm_list_get_ref_assign(outputs, 0, &list_ref);
    if (iree_status_is_ok(status)) {
      *out_steps_list = iree_vm_list_deref(list_ref);
      if (*out_steps_list) {
        iree_vm_list_retain(*out_steps_list);
      }
    }
  }

  iree_vm_list_release(outputs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Execute encoder
//===----------------------------------------------------------------------===//

static iree_status_t iree_encode_execute(iree_vm_context_t* context,
                                         iree_hal_device_t* device,
                                         iree_encode_target_t* target,
                                         iree_vm_list_t* steps_list,
                                         iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build inputs: [steps_list, wait_fence, signal_fence].
  iree_vm_list_t* inputs = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             3, host_allocator, &inputs);

  // Push steps list (may be NULL).
  if (iree_status_is_ok(status)) {
    if (steps_list) {
      iree_vm_ref_t steps_ref = iree_vm_list_retain_ref(steps_list);
      status = iree_vm_list_push_ref_move(inputs, &steps_ref);
    } else {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      status = iree_vm_list_push_ref_move(inputs, &null_ref);
    }
  }

  // Append async fences.
  iree_hal_fence_t* signal_fence = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_tooling_append_async_fences(inputs, target->encode_fn, device,
                                         /*wait_fence=*/NULL, &signal_fence);
  }

  // Invoke encoder.
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        context, target->encode_fn, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, inputs, /*outputs=*/NULL, host_allocator);
  }

  iree_vm_list_release(inputs);

  // Wait for completion.
  if (iree_status_is_ok(status) && signal_fence) {
    status = iree_hal_fence_wait(signal_fence, iree_infinite_timeout(),
                                 IREE_HAL_WAIT_FLAG_DEFAULT);
  }

  iree_hal_fence_release(signal_fence);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Dump output parameters
//===----------------------------------------------------------------------===//

// Dumps the contents of output archives similar to iree-dump-parameters.
static iree_status_t iree_encode_dump_outputs(iree_output_archive_t* archives,
                                              iree_host_size_t archive_count,
                                              iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_builder_t sb;
  iree_string_builder_initialize(host_allocator, &sb);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < archive_count && iree_status_is_ok(status);
       ++i) {
    iree_output_archive_t* archive = &archives[i];
    if (!archive->index) continue;

    status = iree_string_builder_append_cstring(
        &sb,
        "//"
        "===-----------------------------------------------------------------"
        "---------------------------------------------===//\n");
    if (!iree_status_is_ok(status)) break;

    // Print archive header.
    iree_io_physical_size_t archive_size =
        iree_io_parameter_archive_builder_total_size(&archive->builder);
    status = iree_string_builder_append_format(
        &sb, "// Output: %.*s (%" PRIu64 " bytes)\n", (int)archive->path.size,
        archive->path.data, archive_size);
    if (!iree_status_is_ok(status)) break;

    // Dump parameter index.
    status = iree_io_parameter_index_dump(archive->scope, archive->index, &sb);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "%.*s", (int)iree_string_builder_size(&sb),
            iree_string_builder_buffer(&sb));
  }

  iree_string_builder_deinitialize(&sb);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Main encoding workflow
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_encode_parameters(
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();

  // Create VM instance.
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_create_instance(host_allocator, &instance));

  // Load modules and discover encoder functions.
  iree_tooling_module_list_t module_list;
  iree_vm_module_t* encoder_module = NULL;
  iree_encode_target_set_t target_set;
  status = iree_encode_load_and_discover(instance, host_allocator, &module_list,
                                         &encoder_module, &target_set);

  // Select target.
  iree_encode_target_t* selected_target = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_encode_select_target(&target_set, &selected_target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_encode_validate_target(selected_target);
  }

  // Handle --list_targets (early exit).
  if (iree_status_is_ok(status) && FLAG_list_targets) {
    status = iree_encode_print_targets(encoder_module, &target_set);
    iree_encode_target_set_deinitialize(&target_set);
    iree_tooling_module_list_reset(&module_list);
    iree_vm_instance_release(instance);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Call indices function.
  iree_vm_list_t* indices_list = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_encode_call_indices(instance, &module_list, selected_target,
                                      host_allocator, &indices_list);
  }

  // Handle --list_parameters (early exit).
  if (iree_status_is_ok(status) && FLAG_list_parameters) {
    status = iree_encode_print_parameters(indices_list);
    iree_vm_list_release(indices_list);
    iree_encode_target_set_deinitialize(&target_set);
    iree_tooling_module_list_reset(&module_list);
    iree_vm_instance_release(instance);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Parse output flags.
  iree_output_scope_list_t output_list;
  iree_output_scope_list_initialize(host_allocator, &output_list);
  if (iree_status_is_ok(status)) {
    status = iree_encode_parse_output_flags(&output_list);
  }
  if (iree_status_is_ok(status) && output_list.count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no output specified; use --output=[scope=]path.irpa "
        "(e.g., --output=encoded=output.irpa or --output=output.irpa)");
  }

  // Create output archives.
  iree_output_archive_t* archives = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_encode_create_archives(indices_list, &output_list,
                                         host_allocator, &archives);
  }

  // Create encoding context with output providers.
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_encode_create_encoding_context(
        instance, &module_list, archives, output_list.count, host_allocator,
        &context, &device);
  }

  // Call steps function.
  iree_vm_list_t* steps_list = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_encode_call_steps(context, selected_target, host_allocator,
                                    &steps_list);
  }

  // Execute encoder.
  if (iree_status_is_ok(status)) {
    status = iree_encode_execute(context, device, selected_target, steps_list,
                                 host_allocator);
  }

  // Dump output parameters (unless quiet mode).
  if (iree_status_is_ok(status) && !FLAG_quiet) {
    status =
        iree_encode_dump_outputs(archives, output_list.count, host_allocator);
  }

  // Cleanup.
  iree_vm_list_release(steps_list);
  iree_vm_list_release(indices_list);
  if (archives) {
    for (iree_host_size_t i = 0; i < output_list.count; ++i) {
      iree_output_archive_deinitialize(&archives[i]);
    }
    iree_allocator_free(host_allocator, archives);
  }
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_output_scope_list_deinitialize(&output_list);
  iree_encode_target_set_deinitialize(&target_set);
  iree_tooling_module_list_reset(&module_list);
  iree_vm_instance_release(instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-encode-parameters",
      "Encodes parameter files using an encoding module.\n"
      "\n"
      "This tool transforms model parameters using an encoder module produced\n"
      "by iree-compile with --iree-parameter-encoder-output-file. The encoder\n"
      "pre-computes parameter transformations (packing, encoding, dispatches)\n"
      "that would otherwise run at model load time.\n"
      "\n"
      "WORKFLOW:\n"
      "  1. Compile main module with encoder output:\n"
      "       iree-compile model.mlir \\\n"
      "         --iree-parameter-encoder-output-file=encoder.mlir \\\n"
      "         --iree-parameter-splat-path=input.irpa \\\n"
      "         -o main.vmfb\n"
      "\n"
      "  2. Compile the encoder module:\n"
      "       iree-compile encoder.mlir -o encoder.vmfb\n"
      "\n"
      "  3. Run the encoder to transform parameters:\n"
      "       iree-encode-parameters \\\n"
      "         --module=encoder.vmfb \\\n"
      "         --parameters=model=input.irpa \\\n"
      "         --output=encoded=output.irpa\n"
      "\n"
      "  4. Run the main module with encoded parameters:\n"
      "       iree-run-module \\\n"
      "         --module=main.vmfb \\\n"
      "         --parameters=model=input.irpa \\\n"
      "         --parameters=encoded=output.irpa\n"
      "\n"
      "FLAGS:\n"
      "  --module=path.vmfb     Encoder module (required)\n"
      "  --parameters=scope=path  Input parameter file(s)\n"
      "  --output=scope=path.irpa  Output encoded parameter file(s)\n"
      "  --list-targets         List available encoding targets\n"
      "  --list-parameters      List parameters that will be encoded\n"
      "  --target=name          Select specific target (default: auto-detect)\n"
      "  --quiet                Suppress output except errors\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  if (argc > 1) {
    fprintf(stderr, "Error: no positional arguments expected.\n");
    fprintf(stderr,
            "Use one or more --parameters=file.ext flags to specify parameter "
            "files.\n");
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(exit_code);
    return EXIT_FAILURE;
  }

  iree_status_t status = iree_tooling_encode_parameters(host_allocator);

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
