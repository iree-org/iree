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

#include "iree/vm/module.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/tracing.h"
#include "iree/vm/ref.h"

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_function_call_get_cconv_fragments(
    const iree_vm_function_signature_t* signature,
    iree_string_view_t* out_arguments, iree_string_view_t* out_results) {
  memset(out_arguments, 0, sizeof(*out_arguments));
  memset(out_results, 0, sizeof(*out_results));
  iree_string_view_t cconv = signature->calling_convention;
  if (!cconv.size) {
    // No cconv string, so function is `()->()`.
    return iree_ok_status();
  } else if (cconv.data[0] != '0') {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported cconv version %c", cconv.data[0]);
  }
  iree_string_view_t cconv_body = iree_string_view_substr(cconv, 1, INTPTR_MAX);
  if (iree_string_view_split(cconv_body, '.', out_arguments, out_results) ==
      -1) {
    *out_arguments = cconv_body;
  }
  return iree_ok_status();
}

IREE_API_EXPORT bool IREE_API_CALL
iree_vm_function_call_is_variadic_cconv(iree_string_view_t cconv) {
  return iree_string_view_find_char(cconv, IREE_VM_CCONV_TYPE_SPAN_START, 0) !=
         IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_function_call_compute_cconv_fragment_size(
    iree_string_view_t cconv_fragment,
    const iree_vm_register_list_t* segment_size_list,
    iree_host_size_t* out_required_size) {
  iree_host_size_t required_size = 0;
  for (iree_host_size_t i = 0, seg_i = 0; i < cconv_fragment.size;
       ++i, ++seg_i) {
    switch (cconv_fragment.data[i]) {
      case IREE_VM_CCONV_TYPE_INT32:
        required_size += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_INT64:
        required_size += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        required_size += sizeof(iree_vm_ref_t);
        break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        if (IREE_UNLIKELY(!segment_size_list) ||
            IREE_UNLIKELY(seg_i >= segment_size_list->size)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "variadic argument found but segment size "
                                  "list is missing/underflowed");
        }
        iree_host_size_t span_count = segment_size_list->registers[seg_i];
        required_size += sizeof(int32_t);  // count
        iree_host_size_t span_size = 0;
        for (i = i + 1; i < cconv_fragment.size &&
                        cconv_fragment.data[i] != IREE_VM_CCONV_TYPE_SPAN_END;
             ++i) {
          switch (cconv_fragment.data[i]) {
            case IREE_VM_CCONV_TYPE_INT32:
              span_size += sizeof(int32_t);
              break;
            case IREE_VM_CCONV_TYPE_INT64:
              span_size += sizeof(int64_t);
              break;
            case IREE_VM_CCONV_TYPE_REF:
              span_size += sizeof(iree_vm_ref_t);
              break;
            default:
              return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "unsupported cconv span type %c",
                                      cconv_fragment.data[i]);
          }
        }
        required_size += span_size * span_count;
      } break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported cconv type %c",
                                cconv_fragment.data[i]);
    }
  }
  *out_required_size = required_size;
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_function_call_release(iree_vm_function_call_t* call,
                              const iree_vm_function_signature_t* signature) {
  if (!call->arguments.data_length || !call->results.data_length) {
    return;
  }
  iree_string_view_t cconv = signature->calling_convention;
  if (cconv.size == 0 || cconv.data[0] != '0') return;
  uint8_t* p = call->arguments.data;
  for (iree_host_size_t i = 1; i < cconv.size; ++i) {
    char c = cconv.data[i];
    if (c == '.') {
      // Switch to results.
      p = call->results.data;
    }
    switch (c) {
      case IREE_VM_CCONV_TYPE_INT32:
        p += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_INT64:
        p += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        iree_vm_ref_release((iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
        break;
    }
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_initialize(iree_vm_module_t* module, void* self) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(module, 0, sizeof(iree_vm_module_t));
  module->self = self;
  iree_atomic_ref_count_init(&module->ref_count);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_retain(iree_vm_module_t* module) {
  if (module) {
    iree_atomic_ref_count_inc(&module->ref_count);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_release(iree_vm_module_t* module) {
  if (module && iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    module->destroy(module->self);
  }
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_module_name(const iree_vm_module_t* module) {
  if (!module) {
    return iree_make_cstring_view("null");
  }
  return module->name(module->self);
}

IREE_API_EXPORT iree_vm_module_signature_t IREE_API_CALL
iree_vm_module_signature(const iree_vm_module_t* module) {
  if (!module) {
    iree_vm_module_signature_t empty;
    memset(&empty, 0, sizeof(empty));
    return empty;
  }
  return module->signature(module->self);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_lookup_function_by_name(const iree_vm_module_t* module,
                                       iree_vm_function_linkage_t linkage,
                                       iree_string_view_t name,
                                       iree_vm_function_t* out_function) {
  return module->lookup_function(module->self, linkage, name, out_function);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_lookup_function_by_ordinal(const iree_vm_module_t* module,
                                          iree_vm_function_linkage_t linkage,
                                          iree_host_size_t ordinal,
                                          iree_vm_function_t* out_function,
                                          iree_string_view_t* linkage_name) {
  return module->get_function(module->self, linkage, ordinal, out_function,
                              /*out_name=*/linkage_name,
                              /*out_signature=*/NULL);
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_function_name(const iree_vm_function_t* function) {
  iree_string_view_t name;
  iree_status_t status = function->module->get_function(
      function->module->self, function->linkage, function->ordinal,
      /*out_function=*/NULL,
      /*out_name=*/&name,
      /*out_signature=*/NULL);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return iree_make_cstring_view("<error>");
  }
  return name;
}

IREE_API_EXPORT iree_vm_function_signature_t IREE_API_CALL
iree_vm_function_signature(const iree_vm_function_t* function) {
  iree_vm_function_signature_t signature;
  memset(&signature, 0, sizeof(signature));
  IREE_IGNORE_ERROR(function->module->get_function(
      function->module->self, function->linkage, function->ordinal,
      /*out_function=*/NULL,
      /*out_name=*/NULL,
      /*out_signature=*/&signature));
  return signature;
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_function_reflection_attr(const iree_vm_function_t* function,
                                 iree_string_view_t key) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_vm_module_t* module = function->module;
  if (!module->get_function_reflection_attr) {
    IREE_TRACE_ZONE_END(z0);
    return iree_string_view_empty();
  }
  for (int index = 0;; ++index) {
    iree_string_view_t index_key, index_value;
    iree_status_t status = module->get_function_reflection_attr(
        module->self, function->linkage, function->ordinal, index, &index_key,
        &index_value);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }
    if (iree_string_view_compare(key, index_key) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return index_value;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_string_view_empty();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_get_function_reflection_attr(iree_vm_function_t function,
                                     iree_host_size_t index,
                                     iree_string_view_t* key,
                                     iree_string_view_t* value) {
  if (!function.module->get_function_reflection_attr) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "reflection not available for the given module");
  }
  return function.module->get_function_reflection_attr(
      function.module->self, function.linkage, function.ordinal, index, key,
      value);
}
