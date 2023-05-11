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

//===----------------------------------------------------------------------===//
// Function ABI management
//===----------------------------------------------------------------------===//

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
                                      "unsupported cconv span type '%c'",
                                      cconv_fragment.data[i]);
          }
        }
      } break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported cconv type '%c'",
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
                                      "unsupported cconv span type '%c'",
                                      cconv_fragment.data[i]);
          }
        }
        required_size += span_size * span_count;
      } break;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported cconv type '%c'",
                                cconv_fragment.data[i]);
    }
  }
  *out_required_size = required_size;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source locations
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_vm_source_location_format(iree_vm_source_location_t* source_location,
                               iree_vm_source_location_format_flags_t flags,
                               iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  if (!source_location || !source_location->format) {
    return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
  }
  return source_location->format(source_location->self, source_location->data,
                                 flags, builder);
}

//===----------------------------------------------------------------------===//
// iree_vm_module_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(iree_vm_module_t* module, void* self) {
  IREE_ASSERT_ARGUMENT(module);
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

IREE_API_EXPORT iree_string_view_t iree_vm_module_lookup_attr_by_name(
    const iree_vm_module_t* module, iree_string_view_t key) {
  IREE_ASSERT_ARGUMENT(module);
  if (!module->get_module_attr) {
    return iree_string_view_empty();
  }
  for (iree_host_size_t index = 0;; ++index) {
    iree_string_pair_t attr;
    iree_status_t status = module->get_module_attr(module->self, index, &attr);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    } else if (iree_string_view_equal(key, attr.key)) {
      return attr.value;
    }
  }
  return iree_string_view_empty();
}

IREE_API_EXPORT iree_status_t
iree_vm_module_get_attr(const iree_vm_module_t* module, iree_host_size_t index,
                        iree_string_pair_t* out_attr) {
  IREE_ASSERT_ARGUMENT(module);
  if (!module->get_module_attr) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  return module->get_module_attr(module->self, index, out_attr);
}

IREE_API_EXPORT iree_status_t iree_vm_module_enumerate_dependencies(
    iree_vm_module_t* module, iree_vm_module_dependency_callback_t callback,
    void* user_data) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(callback);
  if (!module->enumerate_dependencies) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      module->enumerate_dependencies(module->self, callback, user_data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_name(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_string_view_t name, iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(module);
  return module->lookup_function(module->self, linkage, name,
                                 /*expected_signature=*/NULL, out_function);
}

IREE_API_EXPORT iree_status_t iree_vm_module_lookup_function_by_ordinal(
    const iree_vm_module_t* module, iree_vm_function_linkage_t linkage,
    iree_host_size_t ordinal, iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(module);
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

//===----------------------------------------------------------------------===//
// iree_vm_function_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t
iree_vm_function_name(const iree_vm_function_t* function) {
  IREE_ASSERT_ARGUMENT(function);
  if (!function->module) return iree_string_view_empty();
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
  IREE_ASSERT_ARGUMENT(function);
  iree_vm_function_signature_t signature;
  memset(&signature, 0, sizeof(signature));
  IREE_IGNORE_ERROR(function->module->get_function(
      function->module->self, function->linkage, function->ordinal,
      /*out_function=*/NULL,
      /*out_name=*/NULL,
      /*out_signature=*/&signature));
  return signature;
}

IREE_API_EXPORT iree_string_view_t iree_vm_function_lookup_attr_by_name(
    const iree_vm_function_t* function, iree_string_view_t key) {
  IREE_ASSERT_ARGUMENT(function);
  iree_vm_module_t* module = function->module;
  IREE_ASSERT_ARGUMENT(module);
  if (!module->get_function_attr) {
    return iree_string_view_empty();
  }
  for (iree_host_size_t index = 0;; ++index) {
    iree_string_pair_t attr;
    iree_status_t status = module->get_function_attr(
        module->self, function->linkage, function->ordinal, index, &attr);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    } else if (iree_string_view_equal(key, attr.key)) {
      return attr.value;
    }
  }
  return iree_string_view_empty();
}

IREE_API_EXPORT iree_status_t
iree_vm_function_get_attr(iree_vm_function_t function, iree_host_size_t index,
                          iree_string_pair_t* out_attr) {
  IREE_ASSERT_ARGUMENT(function.module);
  if (!function.module->get_function_attr) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  return function.module->get_function_attr(function.module->self,
                                            function.linkage, function.ordinal,
                                            index, out_attr);
}
