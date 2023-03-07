// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/vm_util.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/numpy_io.h"

static iree_status_t iree_allocate_and_copy_cstring_from_view(
    iree_allocator_t allocator, iree_string_view_t view, char** cstring) {
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, view.size + 1, (void**)cstring));
  memcpy(*cstring, view.data, view.size);
  (*cstring)[view.size] = 0;
  return iree_ok_status();
}

static iree_status_t iree_tooling_load_ndarrays_from_file(
    iree_string_view_t file_path, iree_hal_allocator_t* device_allocator,
    iree_vm_list_t* list) {
  char* file_path_cstring = NULL;
  IREE_RETURN_IF_ERROR(iree_allocate_and_copy_cstring_from_view(
      iree_allocator_system(), file_path, &file_path_cstring));
  FILE* file = fopen(file_path_cstring, "rb");
  iree_allocator_free(iree_allocator_system(), file_path_cstring);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  uint64_t file_length = 0;
  iree_status_t status = iree_file_query_length(file, &file_length);

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  while (iree_status_is_ok(status) && !iree_file_is_at(file, file_length)) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    status = iree_numpy_npy_load_ndarray(
        file, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
        device_allocator, &buffer_view);
    if (iree_status_is_ok(status)) {
      iree_vm_ref_t buffer_view_ref =
          iree_hal_buffer_view_retain_ref(buffer_view);
      status = iree_vm_list_push_ref_move(list, &buffer_view_ref);
    }
    iree_hal_buffer_view_release(buffer_view);
  }

  fclose(file);
  return status;
}

struct iree_create_buffer_from_file_generator_user_data_t {
  FILE* file;
};

static iree_status_t iree_create_buffer_from_file_generator_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  struct iree_create_buffer_from_file_generator_user_data_t* read_params =
      user_data;
  size_t bytes_read = fread(mapping->contents.data, 1,
                            mapping->contents.data_length, read_params->file);
  if (bytes_read != mapping->contents.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file contents truncated; expected %zu bytes "
                            "based on buffer view size",
                            mapping->contents.data_length);
  }
  return iree_ok_status();
}

// Creates a HAL buffer view with the given |metadata| and reads the contents
// from the file at |file_path|.
//
// The file contents are directly read in to memory with no processing.
static iree_status_t iree_create_buffer_view_from_file(
    iree_string_view_t metadata, iree_string_view_t file_path,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;

  // Parse shape and element type used to allocate the buffer view.
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result = iree_hal_parse_shape_and_element_type(
      metadata, 0, &shape_rank, NULL, &element_type);
  if (!iree_status_is_ok(shape_result) &&
      !iree_status_is_out_of_range(shape_result)) {
    return shape_result;
  } else if (shape_rank > 128) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "a shape rank of %zu is just a little bit excessive, eh?", shape_rank);
  }
  iree_status_ignore(shape_result);
  iree_hal_dim_t* shape =
      (iree_hal_dim_t*)iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  IREE_RETURN_IF_ERROR(iree_hal_parse_shape_and_element_type(
      metadata, shape_rank, &shape_rank, shape, &element_type));

  // TODO(benvanik): allow specifying the encoding.
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  // Open the file for reading.
  char* file_path_cstring = NULL;
  IREE_RETURN_IF_ERROR(iree_allocate_and_copy_cstring_from_view(
      iree_allocator_system(), file_path, &file_path_cstring));
  FILE* file = fopen(file_path_cstring, "rb");
  iree_allocator_free(iree_allocator_system(), file_path_cstring);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  struct iree_create_buffer_from_file_generator_user_data_t read_params = {
      file,
  };
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      device_allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params, iree_create_buffer_from_file_generator_callback,
      &read_params, out_buffer_view);

  fclose(file);

  return status;
}

iree_status_t iree_tooling_parse_to_variant_list(
    iree_hal_allocator_t* device_allocator,
    const iree_string_view_t* input_strings,
    iree_host_size_t input_strings_count, iree_allocator_t host_allocator,
    iree_vm_list_t** out_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_list = NULL;
  iree_vm_list_t* list = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_list_create(
          /*element_type=*/NULL, input_strings_count, host_allocator, &list));

  iree_status_t status = iree_tooling_parse_into_variant_list(
      device_allocator, input_strings, input_strings_count, host_allocator,
      list);
  if (iree_status_is_ok(status)) {
    *out_list = list;
  } else {
    iree_vm_list_release(list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_parse_into_variant_list(
    iree_hal_allocator_t* device_allocator,
    const iree_string_view_t* input_strings,
    iree_host_size_t input_strings_count, iree_allocator_t host_allocator,
    iree_vm_list_t* list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the list and prepare for pushing items.
  iree_vm_list_clear(list);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_reserve(list, input_strings_count));

  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < input_strings_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_string_view_t input_view = iree_string_view_trim(input_strings[i]);
    if (iree_string_view_is_empty(input_view)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no value specified for input");
      break;
    } else if (iree_string_view_consume_prefix(&input_view, IREE_SV("@"))) {
      status = iree_tooling_load_ndarrays_from_file(input_view,
                                                    device_allocator, list);
      continue;
    } else if (iree_string_view_equal(input_view, IREE_SV("(null)")) ||
               iree_string_view_equal(input_view, IREE_SV("(ignored)"))) {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      status = iree_vm_list_push_ref_retain(list, &null_ref);
      continue;
    }
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value) or buffer.
      bool is_storage_reference = iree_string_view_consume_prefix(
          &input_view, iree_make_cstring_view("&"));
      iree_hal_buffer_view_t* buffer_view = NULL;
      bool has_at = iree_string_view_find_char(input_view, '@', 0) !=
                    IREE_STRING_VIEW_NPOS;
      if (has_at) {
        // Referencing an external file; split into the portion used to
        // initialize the buffer view and the file contents.
        iree_string_view_t metadata, file_path;
        iree_string_view_split(input_view, '@', &metadata, &file_path);
        iree_string_view_consume_suffix(&metadata, iree_make_cstring_view("="));
        status = iree_create_buffer_view_from_file(
            metadata, file_path, device_allocator, &buffer_view);
        if (!iree_status_is_ok(status)) break;
      } else {
        status = iree_hal_buffer_view_parse(input_view, device_allocator,
                                            &buffer_view);
        if (!iree_status_is_ok(status)) {
          status =
              iree_status_annotate_f(status, "parsing value '%.*s'",
                                     (int)input_view.size, input_view.data);
          break;
        }
      }
      if (is_storage_reference) {
        // Storage buffer reference; just take the storage for the buffer view -
        // it'll still have whatever contents were specified (or 0) but we'll
        // discard the metadata.
        iree_vm_ref_t buffer_ref = iree_hal_buffer_retain_ref(
            iree_hal_buffer_view_buffer(buffer_view));
        iree_hal_buffer_view_release(buffer_view);
        status = iree_vm_list_push_ref_move(list, &buffer_ref);
        if (!iree_status_is_ok(status)) break;
      } else {
        iree_vm_ref_t buffer_view_ref =
            iree_hal_buffer_view_move_ref(buffer_view);
        status = iree_vm_list_push_ref_move(list, &buffer_view_ref);
        if (!iree_status_is_ok(status)) break;
      }
    } else {
      // Scalar.
      bool has_dot = iree_string_view_find_char(input_view, '.', 0) !=
                     IREE_STRING_VIEW_NPOS;
      iree_vm_value_t val;
      if (has_dot) {
        // Float.
        val = iree_vm_value_make_f32(0.0f);
        if (!iree_string_view_atof(input_view, &val.f32)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "parsing value '%.*s' as f32",
                                    (int)input_view.size, input_view.data);
          break;
        }
      } else {
        // Integer.
        val = iree_vm_value_make_i32(0);
        if (!iree_string_view_atoi_int32(input_view, &val.i32)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "parsing value '%.*s' as i32",
                                    (int)input_view.size, input_view.data);
          break;
        }
      }
      status = iree_vm_list_push_value(list, &val);
      if (!iree_status_is_ok(status)) break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_append_async_fence_inputs(
    iree_vm_list_t* list, const iree_vm_function_t* function,
    iree_hal_device_t* device, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t** out_signal_fence) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_view_t model =
      iree_vm_function_lookup_attr_by_name(function, IREE_SV("iree.abi.model"));
  if (!iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
    // Ignore unknown models - the user may have provided their own fences.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Create the signal fence as a 0->1 transition. The caller will wait on that.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_create(device, 0ull, &semaphore));
  iree_hal_fence_t* signal_fence = NULL;
  iree_status_t status = iree_hal_fence_create_at(
      semaphore, 1ull, iree_hal_device_host_allocator(device), &signal_fence);
  iree_hal_semaphore_release(semaphore);

  // Append (wait, signal) fences.
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t wait_fence_ref = iree_hal_fence_retain_ref(wait_fence);
    status = iree_vm_list_push_ref_move(list, &wait_fence_ref);
    iree_vm_ref_release(&wait_fence_ref);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t signal_fence_ref = iree_hal_fence_retain_ref(signal_fence);
    status = iree_vm_list_push_ref_move(list, &signal_fence_ref);
    iree_vm_ref_release(&signal_fence_ref);
  }

  if (iree_status_is_ok(status)) {
    *out_signal_fence = signal_fence;
  } else {
    iree_hal_fence_release(signal_fence);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

#define IREE_PRINTVARIANT_CASE_I(SIZE, B, V)  \
  case IREE_VM_VALUE_TYPE_I##SIZE:            \
    return iree_string_builder_append_format( \
        B, "i" #SIZE "=%" PRIi##SIZE "\n", (V).i##SIZE);

#define IREE_PRINTVARIANT_CASE_F(SIZE, B, V) \
  case IREE_VM_VALUE_TYPE_F##SIZE:           \
    return iree_string_builder_append_format(B, "f" #SIZE "=%g\n", (V).f##SIZE);

// Prints variant description including a trailing newline.
static iree_status_t iree_variant_format(iree_vm_variant_t variant,
                                         iree_host_size_t max_element_count,
                                         iree_string_builder_t* builder) {
  if (iree_vm_variant_is_empty(variant)) {
    return iree_string_builder_append_string(builder, IREE_SV("(null)\n"));
  } else if (iree_vm_variant_is_value(variant)) {
    switch (variant.type.value_type) {
      IREE_PRINTVARIANT_CASE_I(8, builder, variant)
      IREE_PRINTVARIANT_CASE_I(16, builder, variant)
      IREE_PRINTVARIANT_CASE_I(32, builder, variant)
      IREE_PRINTVARIANT_CASE_I(64, builder, variant)
      IREE_PRINTVARIANT_CASE_F(32, builder, variant)
      IREE_PRINTVARIANT_CASE_F(64, builder, variant)
      default:
        return iree_string_builder_append_string(builder, IREE_SV("?\n"));
    }
  } else if (iree_vm_variant_is_ref(variant)) {
    iree_string_view_t type_name = iree_vm_ref_type_name(variant.type.ref_type);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, type_name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
    if (iree_vm_list_isa(variant.ref)) {
      iree_vm_list_t* child_list = iree_vm_list_deref(variant.ref);
      IREE_RETURN_IF_ERROR(iree_tooling_append_variant_list_lines(
          IREE_SV("child_list"), child_list, max_element_count, builder));
      return iree_string_builder_append_string(builder, IREE_SV("\n"));
    } else if (iree_hal_buffer_view_isa(variant.ref)) {
      iree_hal_buffer_view_t* buffer_view =
          iree_hal_buffer_view_deref(variant.ref);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_append_to_builder(
          buffer_view, max_element_count, builder));
      return iree_string_builder_append_string(builder, IREE_SV("\n"));
    } else {
      // TODO(benvanik): a way for ref types to describe themselves.
      return iree_string_builder_append_string(builder,
                                               IREE_SV("(no printer)\n"));
    }
  } else {
    return iree_string_builder_append_string(builder, IREE_SV("(null)\n"));
  }
  return iree_ok_status();
}

static iree_status_t iree_variant_fprint(iree_vm_variant_t variant,
                                         iree_host_size_t max_element_count,
                                         FILE* file) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status =
      iree_variant_format(variant, max_element_count, &builder);
  if (iree_status_is_ok(status)) {
    size_t written = fwrite(iree_string_builder_buffer(&builder), 1,
                            iree_string_builder_size(&builder), file);
    if (written != iree_string_builder_size(&builder)) {
      status = iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
    }
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t iree_tooling_append_variant_list_lines(
    iree_string_view_t list_name, iree_vm_list_t* list,
    iree_host_size_t max_element_count, iree_string_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_variant_assign(list, i, &variant),
        "variant %zu not present", i);
    iree_string_builder_append_format(
        builder, "%.*s[%" PRIhsz "]: ", (int)list_name.size, list_name.data, i);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_variant_format(variant, max_element_count, builder));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_tooling_variant_list_fprint(
    iree_string_view_t list_name, iree_vm_list_t* list,
    iree_host_size_t max_element_count, FILE* file) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_tooling_append_variant_list_lines(
      list_name, list, max_element_count, &builder);
  if (iree_status_is_ok(status)) {
    size_t written = fwrite(iree_string_builder_buffer(&builder), 1,
                            iree_string_builder_size(&builder), file);
    if (written != iree_string_builder_size(&builder)) {
      status = iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
    }
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t iree_tooling_output_variant(
    iree_vm_variant_t variant, iree_string_view_t output_str,
    iree_host_size_t max_element_count, FILE* default_file) {
  if (iree_string_view_is_empty(output_str)) {
    // Send into the void.
    return iree_ok_status();
  } else if (iree_string_view_equal(output_str, IREE_SV("-"))) {
    // Route to the provided file.
    return iree_variant_fprint(variant, max_element_count, default_file);
  }

  bool has_at = iree_string_view_consume_prefix(&output_str, IREE_SV("@"));
  bool has_plus = iree_string_view_consume_prefix(&output_str, IREE_SV("+"));
  if (!has_at && !has_plus) {
    // Other types of outputs are not yet supported. We could allow for shapes
    // and either verify metadata or output binary files ala
    // `--input=4xf32=@foo.bin`.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported output mode specification '%.*s'",
                            (int)output_str.size, output_str.data);
  }

  // For now we just send buffer views to npy files as primitive values (like
  // just a normal int) can't be round-tripped. We could wrap the primitives in
  // a single-element buffer view if needed.
  if (!iree_vm_variant_is_ref(variant) ||
      !iree_hal_buffer_view_isa(variant.ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only buffer views can be written to npy files");
  }
  iree_hal_buffer_view_t* buffer_view = iree_hal_buffer_view_deref(variant.ref);

  // Open file for either overwriting or appending (npy files can contain
  // multiple arrays).
  iree_string_view_t file_path = output_str;
  char* file_path_cstring = NULL;
  IREE_RETURN_IF_ERROR(iree_allocate_and_copy_cstring_from_view(
      iree_allocator_system(), file_path, &file_path_cstring));
  const char* mode = has_plus ? "ab" : "wb";
  FILE* file = fopen(file_path_cstring, mode);
  iree_allocator_free(iree_allocator_system(), file_path_cstring);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  // Append buffer view contents to the file stream.
  iree_numpy_npy_save_options_t options = IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT;
  iree_status_t status = iree_numpy_npy_save_ndarray(file, options, buffer_view,
                                                     iree_allocator_system());

  fclose(file);
  return status;
}

iree_status_t iree_tooling_output_variant_list(
    iree_vm_list_t* list, const iree_string_view_t* output_strings,
    iree_host_size_t output_strings_count, iree_host_size_t max_element_count,
    FILE* file) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(!output_strings_count || output_strings);

  // We only care if there are not enough outputs to satisfy the user
  // request. We could force users to specify all outputs to make this a bit
  // harder to misuse but saving off outputs is a power-user feature.
  if (iree_vm_list_size(list) != output_strings_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%" PRIhsz " outputs specified but the provided list only has %" PRIhsz
        " elements",
        output_strings_count, iree_vm_list_size(list));
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < output_strings_count; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_variant_assign(list, i, &variant));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tooling_output_variant(variant, output_strings[i],
                                        max_element_count, file));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
