// Copyright 2020 Google LLC
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

#include "iree/tools/utils/vm_util.h"

#include <ostream>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/bytecode_module.h"

namespace iree {

Status GetFileContents(const char* path, std::string* out_contents) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_contents = std::string();
  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s'", path);
  }
  iree_status_t status = iree_ok_status();
  if (fseek(file, 0, SEEK_END) == -1) {
    status = iree_make_status(iree_status_code_from_errno(errno), "seek (end)");
  }
  size_t file_size = 0;
  if (iree_status_is_ok(status)) {
    file_size = ftell(file);
    if (file_size == -1L) {
      status =
          iree_make_status(iree_status_code_from_errno(errno), "size query");
    }
  }
  if (iree_status_is_ok(status)) {
    if (fseek(file, 0, SEEK_SET) == -1) {
      status =
          iree_make_status(iree_status_code_from_errno(errno), "seek (beg)");
    }
  }
  std::string contents;
  if (iree_status_is_ok(status)) {
    contents.resize(file_size);
    if (fread((char*)contents.data(), file_size, 1, file) != 1) {
      status =
          iree_make_status(iree_status_code_from_errno(errno),
                           "unable to read entire file contents of '%s'", path);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_contents = std::move(contents);
  }
  fclose(file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

Status ParseToVariantList(iree_hal_allocator_t* allocator,
                          absl::Span<const absl::string_view> input_strings,
                          iree_vm_list_t** out_list) {
  *out_list = NULL;
  vm::ref<iree_vm_list_t> variant_list;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(/*element_type=*/nullptr, input_strings.size(),
                          iree_allocator_system(), &variant_list));
  for (size_t i = 0; i < input_strings.size(); ++i) {
    iree_string_view_t input_view = iree_string_view_trim(iree_make_string_view(
        input_strings[i].data(), input_strings[i].size()));
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value).
      iree_hal_buffer_view_t* buffer_view = nullptr;
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_parse(input_view, allocator, &buffer_view),
          "parsing value '%.*s'", (int)input_view.size, input_view.data);
      auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
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

Status ParseToVariantList(iree_hal_allocator_t* allocator,
                          absl::Span<const std::string> input_strings,
                          iree_vm_list_t** out_list) {
  std::vector<absl::string_view> input_views(input_strings.size());
  for (int i = 0; i < input_strings.size(); ++i) {
    input_views[i] = input_strings[i];
  }
  return ParseToVariantList(allocator, input_views, out_list);
}

Status PrintVariantList(iree_vm_list_t* variant_list, std::ostream* os) {
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
          status = iree_hal_buffer_view_format(
              buffer_view, /*max_element_count=*/1024, result_str.size() + 1,
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

Status CreateHalModule(iree_hal_device_t* device,
                       iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(device, iree_allocator_system(), out_module),
      "creating HAL module");
  return OkStatus();
}

Status LoadBytecodeModule(absl::string_view module_data,
                          iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_create(
          iree_const_byte_span_t{
              reinterpret_cast<const uint8_t*>(module_data.data()),
              module_data.size()},
          iree_allocator_null(), iree_allocator_system(), out_module),
      "deserializing module");
  return OkStatus();
}
}  // namespace iree
