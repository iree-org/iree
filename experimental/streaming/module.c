// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "experimental/streaming/internal.h"
#include "iree/io/file_handle.h"

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_module_extract_metadata(
    iree_hal_streaming_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query the number of exported functions.
  module->symbol_count = iree_hal_executable_export_count(module->executable);
  if (module->symbol_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate storage for export infos and per-symbol op counts together.
  // We want to query the export info once and reuse it as we process. In order
  // to allocate the minimum amount of memory we need to precalculate the
  // required number of unpack operations. Once we do that we avoid
  // recalculating later by caching the results.
  typedef struct op_counts_t {
    uint16_t copy_count;
    uint16_t resolve_count;
  } op_counts_t;
  const iree_host_size_t export_infos_size =
      module->symbol_count * sizeof(iree_hal_executable_export_info_t);
  const iree_host_size_t op_counts_size =
      module->symbol_count * sizeof(op_counts_t);
  uint8_t* temp_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(module->host_allocator,
                                export_infos_size + op_counts_size,
                                (void**)&temp_buffer));
  iree_hal_executable_export_info_t* export_infos =
      (iree_hal_executable_export_info_t*)temp_buffer;
  op_counts_t* symbol_op_counts =
      (op_counts_t*)(temp_buffer + export_infos_size);

  // Count all parameters in all exports so we can allocate one buffer to
  // fetch them all. This is somewhat wasteful as we'll be allocating quite a
  // bit but is easier to see in traces.
  iree_status_t status = iree_ok_status();
  iree_host_size_t total_parameter_count = 0;
  for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
    status = iree_hal_executable_export_info(
        module->executable, (iree_hal_executable_export_ordinal_t)i,
        &export_infos[i]);
    if (!iree_status_is_ok(status)) break;
    total_parameter_count += export_infos[i].parameter_count;
  }

  // Allocate the scratch space for querying parameter info.
  iree_hal_executable_export_parameter_t* parameters = NULL;
  if (iree_status_is_ok(status) && total_parameter_count > 0) {
    status = iree_allocator_malloc(module->host_allocator,
                                   total_parameter_count * sizeof(*parameters),
                                   (void**)&parameters);
  }

  // Analyze each export to determine operation counts.
  // We count the total operations per symbol with copy coalescing.
  iree_host_size_t total_ops = 0;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    const iree_host_size_t parameter_count = export_infos[i].parameter_count;
    if (!parameter_count) continue;
    // Query parameters to analyze coalescing opportunities.
    status = iree_hal_executable_export_parameters(
        module->executable, (iree_hal_executable_export_ordinal_t)i,
        parameter_count, &parameters[parameter_base]);
    if (!iree_status_is_ok(status)) break;
    uint32_t src_offset = 0;
    int32_t last_constant_end = -1;
    for (uint16_t j = 0; j < parameter_count; ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      if (parameter->type ==
          IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING) {
        ++symbol_op_counts[i].resolve_count;
        ++total_ops;
        src_offset += parameter->size;
        last_constant_end = -1;  // break contiguity
      } else {
        // CONSTANT or BUFFER_PTR - check for contiguity.
        // Calculate source offset based on parameter order and sizes.
        if (src_offset != last_constant_end) {
          // New copy operation needed.
          ++symbol_op_counts[i].copy_count;
          ++total_ops;
        }
        src_offset += parameter->size;
        last_constant_end = src_offset;
      }
    }
    parameter_base += parameter_count;
  }

  // Allocate all permanent storage in a single block.
  // Memory layout: [Symbol Array][Symbol0 ops][Symbol1 ops]...
  const iree_host_size_t symbols_size =
      module->symbol_count * sizeof(iree_hal_streaming_symbol_t);
  const iree_host_size_t ops_size =
      total_ops * sizeof(iree_hal_streaming_parameter_op_t);
  const iree_host_size_t total_size = symbols_size + ops_size;
  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(module->host_allocator, total_size,
                                   (void**)&buffer);
  }
  module->symbols = (iree_hal_streaming_symbol_t*)buffer;
  iree_hal_streaming_parameter_op_t* ops_base =
      (iree_hal_streaming_parameter_op_t*)(buffer + symbols_size);

  iree_hal_streaming_parameter_op_t* current_ops = ops_base;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
    symbol->module = module;
    symbol->name = export_infos[i].name;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol->export_ordinal = (iree_hal_executable_export_ordinal_t)i;

    // Function attributes - TODO: Query from export metadata when available.
    // TODO(benvanik): populate from occupancy_info when available.
    symbol->occupancy_info = export_infos[i].occupancy_info;
    symbol->max_threads_per_block = 1024;       // TODO: from metadata.
    symbol->shared_size_bytes = 0;              // TODO: from metadata.
    symbol->local_size_bytes = 0;               // TODO: from metadata.
    symbol->num_regs = 32;                      // TODO: from metadata.
    symbol->max_dynamic_shared_size_bytes = 0;  // TODO: from metadata.

    // Initialize parameter info.
    iree_hal_streaming_parameter_info_t* parameter_info = &symbol->parameters;
    parameter_info->constant_bytes =
        export_infos[i].constant_count * sizeof(uint32_t);
    parameter_info->binding_count = export_infos[i].binding_count;
    parameter_info->copy_count = symbol_op_counts[i].copy_count;
    parameter_info->ops = current_ops;
    const uint16_t parameter_count = export_infos[i].parameter_count;
    if (parameter_count == 0) {
      // No parameters.
      parameter_info->buffer_size = 0;
      continue;
    }

    // Build operations with coalescing.
    // Copy ops go first, then resolve ops.
    uint16_t src_offset = 0;
    uint16_t buffer_size = 0;
    iree_hal_streaming_parameter_op_t* copy_ops_start = current_ops;
    iree_hal_streaming_parameter_op_t* resolve_ops_start =
        current_ops + symbol_op_counts[i].copy_count;
    uint16_t copy_count = 0;
    uint16_t resolve_count = 0;
    iree_hal_streaming_parameter_copy_op_t* active_copy = NULL;
    for (uint16_t j = 0; j < parameter_count; ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      if (parameter->type ==
          IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING) {
        // Update offsets. Bindings are passed as pointers.
        iree_hal_streaming_parameter_resolve_op_t* op =
            &resolve_ops_start[resolve_count].resolve;
        op->src_offset = src_offset;
        op->dst_ordinal = resolve_count;  // binding ordinal
        src_offset += parameter->size;
        buffer_size = src_offset;
        ++resolve_count;
        active_copy = NULL;  // break any active copy operation
      } else {
        // CONSTANT or BUFFER_PTR - try to coalesce and choose offsets.
        if (active_copy &&
            active_copy->src_offset + active_copy->size == src_offset) {
          // Extend the current copy operation.
          active_copy->size += parameter->size;
        } else {
          // Start a new copy operation.
          iree_hal_streaming_parameter_copy_op_t* op =
              &copy_ops_start[copy_count].copy;
          op->size = parameter->size;
          op->src_offset = src_offset;
          op->dst_offset = parameter->offset;  // offset in constants
          ++copy_count;
          active_copy = op;
        }
        src_offset += parameter->size;
        buffer_size = src_offset;
      }
    }
    parameter_info->buffer_size = buffer_size;

    // Advance to next symbol's ops.
    parameter_base += parameter_count;
    current_ops += copy_count + resolve_count;
  }

  iree_allocator_free(module->host_allocator, parameters);
  iree_allocator_free(module->host_allocator, temp_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_module_destroy(
    iree_hal_streaming_module_t* module);

iree_status_t iree_hal_streaming_module_create_from_memory(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t image, iree_allocator_t host_allocator,
    iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(image.data);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Attempt to infer the file format and size.
  // A good API would take that in as otherwise we're trusting arbitrary user
  // data.
  iree_const_byte_span_t executable_data = image;
  char executable_format[64];
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_cache_infer_format(
              context->executable_cache, caching_mode, executable_data,
              sizeof(executable_format), executable_format,
              &executable_data.data_length));

  // Allocate module structure.
  iree_hal_streaming_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->cache = NULL;
  module->executable = NULL;
  module->symbols = NULL;
  module->symbol_count = 0;
  module->file_mapping = NULL;
  module->context = context;
  iree_hal_streaming_context_retain(context);
  module->host_allocator = host_allocator;

  // Use the context's executable cache.
  module->cache = context->executable_cache;
  iree_hal_executable_cache_retain(module->cache);

  // Create HAL executable from binary.
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.caching_mode = caching_mode;
  params.executable_format = iree_make_cstring_view(executable_format);
  params.executable_data = executable_data;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      module->cache, &params, &module->executable);

  // Extract kernel metadata.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_extract_metadata(module);
  }

  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    iree_hal_streaming_module_destroy(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_module_create_from_file(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_caching_mode_t caching_mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open the file for reading.
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, path, host_allocator,
                                   &file_handle));

  // Map the entire file for read access.
  iree_io_file_mapping_t* file_mapping = NULL;
  iree_status_t status = iree_io_file_map_view(
      file_handle, IREE_IO_FILE_ACCESS_READ, 0, IREE_HOST_SIZE_MAX,
      IREE_IO_FILE_MAPPING_FLAG_NONE, host_allocator, &file_mapping);

  // Release the file handle (mapping retains it).
  iree_io_file_handle_release(file_handle);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Get the read-only contents of the mapping.
  iree_const_byte_span_t image = iree_io_file_mapping_contents_ro(file_mapping);

  // Create the module from the mapped memory.
  iree_hal_streaming_module_t* module = NULL;
  status = iree_hal_streaming_module_create_from_memory(
      context,
      caching_mode | IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
      image, host_allocator, &module);

  if (iree_status_is_ok(status)) {
    module->file_mapping = file_mapping;
    *out_module = module;
  } else {
    iree_io_file_mapping_release(file_mapping);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_module_destroy(
    iree_hal_streaming_module_t* module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = module->host_allocator;

  // Release file mapping if present.
  iree_io_file_mapping_release(module->file_mapping);

  // Release symbol metadata.
  iree_allocator_free(module->host_allocator, module->symbols);

  // Release executable.
  iree_hal_executable_release(module->executable);

  // Release executable cache.
  iree_hal_executable_cache_release(module->cache);

  // Release context.
  iree_hal_streaming_context_release(module->context);

  // Free module memory.
  iree_allocator_free(host_allocator, module);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_module_retain(iree_hal_streaming_module_t* module) {
  if (module) {
    iree_atomic_ref_count_inc(&module->ref_count);
  }
}

void iree_hal_streaming_module_release(iree_hal_streaming_module_t* module) {
  if (module && iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    iree_hal_streaming_module_destroy(module);
  }
}

iree_status_t iree_hal_streaming_module_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_type_t expected_type,
    iree_hal_streaming_symbol_t** out_symbol) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_symbol);
  *out_symbol = NULL;

  iree_string_view_t name_view =
      iree_string_view_trim(iree_make_cstring_view(name));
  for (uint32_t i = 0; i < module->symbol_count; ++i) {
    if (iree_string_view_equal(module->symbols[i].name, name_view)) {
      // Check if the symbol type matches expected type.
      if (module->symbols[i].type == expected_type) {
        // Return symbol info as pointer.
        *out_symbol = &module->symbols[i];
        return iree_ok_status();
      } else {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol '%.*s' found but type mismatch (expected %d, got %d)",
            (int)name_view.size, name_view.data, expected_type,
            module->symbols[i].type);
      }
      break;
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "symbol '%.*s' not found in module",
                          (int)name_view.size, name_view.data);
}

iree_status_t iree_hal_streaming_module_function(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_function) {
  return iree_hal_streaming_module_symbol(
      module, name, IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION, out_function);
}

iree_status_t iree_hal_streaming_module_global(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_deviceptr_t* out_device_ptr,
    iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = 0;
  if (out_size) *out_size = 0;

  iree_hal_streaming_symbol_t* symbol = NULL;
  iree_status_t status = iree_hal_streaming_module_symbol(
      module, name, IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL, &symbol);

  if (iree_status_is_ok(status)) {
    *out_device_ptr = symbol->device_address;
    if (out_size) *out_size = symbol->size_bytes;
  }
  return status;
}
