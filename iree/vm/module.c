// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/tracing.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

IREE_API_EXPORT iree_status_t iree_vm_function_call_get_cconv_fragments(
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
  if (iree_string_view_split(cconv_body, '_', out_arguments, out_results) ==
      -1) {
    *out_arguments = cconv_body;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_function_call_count_fragment_values(
    iree_string_view_t cconv_fragment, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < cconv_fragment.size; ++i) {
    switch (cconv_fragment.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
      case IREE_VM_CCONV_TYPE_REF:
        ++count;
        break;
      case IREE_VM_CCONV_TYPE_SPAN_START: {
        for (i = i + 1; i < cconv_fragment.size &&
                        cconv_fragment.data[i] != IREE_VM_CCONV_TYPE_SPAN_END;
             ++i) {
          switch (cconv_fragment.data[i]) {
            case IREE_VM_CCONV_TYPE_VOID:
              break;
            case IREE_VM_CCONV_TYPE_I32:
            case IREE_VM_CCONV_TYPE_F32:
            case IREE_VM_CCONV_TYPE_I64:
            case IREE_VM_CCONV_TYPE_F64:
            case IREE_VM_CCONV_TYPE_REF:
              ++count;
              break;
            default:
              return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "unsupported cconv span type %c",
                                      cconv_fragment.data[i]);
          }
        }
      } break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported cconv type %c",
                                cconv_fragment.data[i]);
    }
  }
  *out_count = count;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_call_count_arguments_and_results(
    const iree_vm_function_signature_t* signature,
    iree_host_size_t* out_argument_count, iree_host_size_t* out_result_count) {
  IREE_ASSERT_ARGUMENT(signature);
  IREE_ASSERT_ARGUMENT(out_argument_count);
  IREE_ASSERT_ARGUMENT(out_result_count);
  *out_argument_count = 0;
  *out_result_count = 0;
  iree_string_view_t arguments, results;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      signature, &arguments, &results));
  IREE_RETURN_IF_ERROR(iree_vm_function_call_count_fragment_values(
      arguments, out_argument_count));
  IREE_RETURN_IF_ERROR(
      iree_vm_function_call_count_fragment_values(results, out_result_count));
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_vm_function_call_is_variadic_cconv(
    iree_string_view_t cconv) {
  return iree_string_view_find_char(cconv, IREE_VM_CCONV_TYPE_SPAN_START, 0) !=
         IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_status_t iree_vm_function_call_compute_cconv_fragment_size(
    iree_string_view_t cconv_fragment,
    const iree_vm_register_list_t* segment_size_list,
    iree_host_size_t* out_required_size) {
  iree_host_size_t required_size = 0;
  for (iree_host_size_t i = 0, seg_i = 0; i < cconv_fragment.size;
       ++i, ++seg_i) {
    switch (cconv_fragment.data[i]) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
        required_size += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
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
            case IREE_VM_CCONV_TYPE_VOID:
              break;
            case IREE_VM_CCONV_TYPE_I32:
            case IREE_VM_CCONV_TYPE_F32:
              span_size += sizeof(int32_t);
              break;
            case IREE_VM_CCONV_TYPE_I64:
            case IREE_VM_CCONV_TYPE_F64:
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

IREE_API_EXPORT void iree_vm_function_call_release(
    iree_vm_function_call_t* call,
    const iree_vm_function_signature_t* signature) {
  if (!call->arguments.data_length || !call->results.data_length) {
    return;
  }
  iree_string_view_t cconv = signature->calling_convention;
  if (cconv.size == 0 || cconv.data[0] != '0') return;
  uint8_t* p = call->arguments.data;
  for (iree_host_size_t i = 1; i < cconv.size; ++i) {
    char c = cconv.data[i];
    if (c == '_') {
      // Switch to results.
      p = call->results.data;
    }
    switch (c) {
      case IREE_VM_CCONV_TYPE_VOID:
        break;
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_F32:
        p += sizeof(int32_t);
        break;
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F64:
        p += sizeof(int64_t);
        break;
      case IREE_VM_CCONV_TYPE_REF:
        iree_vm_ref_release((iree_vm_ref_t*)p);
        p += sizeof(iree_vm_ref_t);
        break;
    }
  }
}

IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(iree_vm_module_t* module, void* self) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(module, 0, sizeof(iree_vm_module_t));
  module->self = self;
  iree_atomic_ref_count_init(&module->ref_count);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_module_retain(iree_vm_module_t* module) {
  if (module) {
    iree_atomic_ref_count_inc(&module->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_module_release(iree_vm_module_t* module) {
  if (module && iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    module->destroy(module->self);
  }
}

IREE_API_EXPORT iree_string_view_t
iree_vm_module_name(const iree_vm_module_t* module) {
  if (!module) {
    return iree_make_cstring_view("null");
  }
  return module->name(module->self);
}

IREE_API_EXPORT iree_vm_module_signature_t
iree_vm_module_signature(const iree_vm_module_t* module) {
  if (!module) {
    iree_vm_module_signature_t empty;
    memset(&empty, 0, sizeof(empty));
    return empty;
  }
  return module->signature(module->self);
}

IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_name(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_string_view_t name, iree_vm_function_t* out_function) {
  return module->lookup_function(module->self, linkage, name, out_function);
}

IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_ordinal(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_host_size_t ordinal, iree_vm_function_t* out_function) {
  return module->get_function(module->self, linkage, ordinal, out_function,
                              /*out_name=*/NULL,
                              /*out_signature=*/NULL);
}

IREE_API_EXPORT iree_status_t iree_vm_module_resolve_source_location(
    const iree_vm_module_t* module, iree_vm_stack_frame_t* frame,
    iree_vm_source_location_t* out_source_location) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(out_source_location);
  memset(out_source_location, 0, sizeof(*out_source_location));
  if (module->resolve_source_location) {
    return module->resolve_source_location(module->self, frame,
                                           out_source_location);
  }
  return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
}

IREE_API_EXPORT iree_status_t
iree_vm_source_location_format(iree_vm_source_location_t* source_location,
                               iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  if (!source_location || !source_location->format) {
    return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
  }
  return source_location->format(source_location->self, source_location->data,
                                 builder);
}

IREE_API_EXPORT iree_string_view_t
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

IREE_API_EXPORT iree_vm_function_signature_t
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

IREE_API_EXPORT iree_string_view_t iree_vm_function_reflection_attr(
    const iree_vm_function_t* function, iree_string_view_t key) {
  iree_vm_module_t* module = function->module;
  if (!module->get_function_reflection_attr) {
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
    if (iree_string_view_equal(key, index_key)) {
      return index_value;
    }
  }
  return iree_string_view_empty();
}

IREE_API_EXPORT iree_status_t iree_vm_get_function_reflection_attr(
    iree_vm_function_t function, iree_host_size_t index,
    iree_string_view_t* key, iree_string_view_t* value) {
  if (!function.module->get_function_reflection_attr) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "reflection not available for the given module");
  }
  return function.module->get_function_reflection_attr(
      function.module->self, function.linkage, function.ordinal, index, key,
      value);
}
