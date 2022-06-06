// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/utils/vm_util.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <type_traits>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/base/status_cc.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/ref_cc.h"

namespace iree {

// Creates a HAL buffer view with the given |metadata| and reads the contents
// from the file at |file_path|.
//
// The file contents are directly read in to memory with no processing.
static iree_status_t CreateBufferViewFromFile(
    iree_string_view_t metadata, iree_string_view_t file_path,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;

  // Parse shape and element type used to allocate the buffer view.
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result = iree_hal_parse_shape_and_element_type(
      metadata, 0, NULL, &shape_rank, &element_type);
  if (!iree_status_is_ok(shape_result) &&
      !iree_status_is_out_of_range(shape_result)) {
    return shape_result;
  } else if (shape_rank > 128) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "a shape rank of %zu is just a little bit excessive, eh?", shape_rank);
  }
  shape_result = iree_status_ignore(shape_result);
  iree_hal_dim_t* shape =
      (iree_hal_dim_t*)iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  IREE_RETURN_IF_ERROR(iree_hal_parse_shape_and_element_type(
      metadata, shape_rank, shape, &shape_rank, &element_type));

  // TODO(benvanik): allow specifying the encoding.
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  // Open the file for reading.
  std::string file_path_str(file_path.data, file_path.size);
  FILE* file = std::fopen(file_path_str.c_str(), "rb");
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%.*s'", (int)file_path.size,
                            file_path.data);
  }

  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER;
  struct read_params_t {
    FILE* file;
  } read_params = {
      file,
  };
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      device_allocator, shape, shape_rank, element_type, encoding_type,
      buffer_params,
      +[](iree_hal_buffer_mapping_t* mapping, void* user_data) {
        auto* read_params = reinterpret_cast<read_params_t*>(user_data);
        size_t bytes_read =
            std::fread(mapping->contents.data, 1, mapping->contents.data_length,
                       read_params->file);
        if (bytes_read != mapping->contents.data_length) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "file contents truncated; expected %zu bytes "
                                  "based on buffer view size",
                                  mapping->contents.data_length);
        }
        return iree_ok_status();
      },
      &read_params, out_buffer_view);

  std::fclose(file);

  return status;
}

Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                          iree::span<const std::string> input_strings,
                          iree_vm_list_t** out_list) {
  *out_list = NULL;
  vm::ref<iree_vm_list_t> variant_list;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      /*element_type=*/nullptr, input_strings.size(),
      iree_hal_allocator_host_allocator(device_allocator), &variant_list));
  for (size_t i = 0; i < input_strings.size(); ++i) {
    iree_string_view_t input_view = iree_string_view_trim(iree_make_string_view(
        input_strings[i].data(), input_strings[i].size()));
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value) or buffer.
      bool is_storage_reference = iree_string_view_consume_prefix(
          &input_view, iree_make_cstring_view("&"));
      iree_hal_buffer_view_t* buffer_view = nullptr;
      bool has_at = iree_string_view_find_char(input_view, '@', 0) !=
                    IREE_STRING_VIEW_NPOS;
      if (has_at) {
        // Referencing an external file; split into the portion used to
        // initialize the buffer view and the file contents.
        iree_string_view_t metadata, file_path;
        iree_string_view_split(input_view, '@', &metadata, &file_path);
        iree_string_view_consume_suffix(&metadata, iree_make_cstring_view("="));
        IREE_RETURN_IF_ERROR(CreateBufferViewFromFile(
            metadata, file_path, device_allocator, &buffer_view));
      } else {
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
                                 input_view, device_allocator, &buffer_view),
                             "parsing value '%.*s'", (int)input_view.size,
                             input_view.data);
      }
      if (is_storage_reference) {
        // Storage buffer reference; just take the storage for the buffer view -
        // it'll still have whatever contents were specified (or 0) but we'll
        // discard the metadata.
        auto buffer_ref = iree_hal_buffer_retain_ref(
            iree_hal_buffer_view_buffer(buffer_view));
        iree_hal_buffer_view_release(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_ref));
      } else {
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
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
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as f32",
                                  (int)input_view.size, input_view.data);
        }
      } else {
        // Integer.
        val = iree_vm_value_make_i32(0);
        if (!iree_string_view_atoi_int32(input_view, &val.i32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as i32",
                                  (int)input_view.size, input_view.data);
        }
      }
      IREE_RETURN_IF_ERROR(iree_vm_list_push_value(variant_list.get(), &val));
    }
  }
  *out_list = variant_list.release();
  return OkStatus();
}

Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        std::ostream* os) {
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variant_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(variant_list, i, &variant),
                         "variant %zu not present", i);

    *os << "result[" << i << "]: ";
    if (iree_vm_variant_is_value(variant)) {
      switch (variant.type.value_type) {
        case IREE_VM_VALUE_TYPE_I8:
          *os << "i8=" << variant.i8 << "\n";
          break;
        case IREE_VM_VALUE_TYPE_I16:
          *os << "i16=" << variant.i16 << "\n";
          break;
        case IREE_VM_VALUE_TYPE_I32:
          *os << "i32=" << variant.i32 << "\n";
          break;
        case IREE_VM_VALUE_TYPE_I64:
          *os << "i64=" << variant.i64 << "\n";
          break;
        case IREE_VM_VALUE_TYPE_F32:
          *os << "f32=" << variant.f32 << "\n";
          break;
        case IREE_VM_VALUE_TYPE_F64:
          *os << "f64=" << variant.f64 << "\n";
          break;
        default:
          *os << "?\n";
          break;
      }
    } else if (iree_vm_variant_is_ref(variant)) {
      iree_string_view_t type_name =
          iree_vm_ref_type_name(variant.type.ref_type);
      *os << std::string(type_name.data, type_name.size) << "\n";
      if (iree_hal_buffer_view_isa(variant.ref)) {
        auto* buffer_view = iree_hal_buffer_view_deref(variant.ref);
        std::string result_str(4096, '\0');
        iree_status_t status;
        do {
          iree_host_size_t actual_length = 0;
          status = iree_hal_buffer_view_format(buffer_view, max_element_count,
                                               result_str.size() + 1,
                                               &result_str[0], &actual_length);
          result_str.resize(actual_length);
        } while (iree_status_is_out_of_range(status));
        IREE_RETURN_IF_ERROR(status);
        *os << result_str << "\n";
      } else {
        // TODO(benvanik): a way for ref types to describe themselves.
        *os << "(no printer)\n";
      }
    } else {
      *os << "(null)\n";
    }
  }

  return OkStatus();
}

Status CreateDevice(const char* driver_name, iree_hal_device_t** out_device) {
  IREE_LOG(INFO) << "Creating driver and device for '" << driver_name << "'...";
  iree_hal_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
                           iree_hal_driver_registry_default(),
                           iree_make_cstring_view(driver_name),
                           iree_allocator_system(), &driver),
                       "creating driver '%s'", driver_name);
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
                           driver, iree_allocator_system(), out_device),
                       "creating default device for driver '%s'", driver_name);
  iree_hal_driver_release(driver);
  return OkStatus();
}

}  // namespace iree
