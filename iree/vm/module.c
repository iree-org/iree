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

#include "iree/base/atomics.h"

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_module_initialize(iree_vm_module_t* module, void* self) {
  memset(module, 0, sizeof(iree_vm_module_t));
  module->self = self;
  iree_atomic_store(&module->ref_count, 1);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_retain(iree_vm_module_t* module) {
  if (module) {
    iree_atomic_fetch_add(&module->ref_count, 1);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_vm_module_release(iree_vm_module_t* module) {
  if (module && iree_atomic_fetch_sub(&module->ref_count, 1) == 1) {
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
                                          int32_t ordinal,
                                          iree_vm_function_t* out_function,
                                          iree_string_view_t* linkage_name) {
  return module->get_function(module->self, linkage, ordinal, out_function,
                              /*out_name=*/linkage_name,
                              /*out_signature=*/NULL);
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_vm_function_name(const iree_vm_function_t* function) {
  iree_string_view_t name;
  if (!iree_status_is_ok(function->module->get_function(
          function->module->self, function->linkage, function->ordinal,
          /*out_function=*/NULL,
          /*out_name=*/&name,
          /*out_signature=*/NULL))) {
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
  iree_string_view_t empty_string = IREE_STRING_VIEW_EMPTY;
  iree_vm_module_t* module = function->module;
  if (!module->get_function_reflection_attr) {
    return empty_string;
  }
  for (int index = 0;; ++index) {
    iree_string_view_t index_key, index_value;
    iree_status_t status = module->get_function_reflection_attr(
        module->self, function->linkage, function->ordinal, index, &index_key,
        &index_value);
    if (!iree_status_is_ok(status)) break;
    if (iree_string_view_compare(key, index_key) == 0) {
      return index_value;
    }
  }
  return empty_string;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_vm_get_function_reflection_attr(iree_vm_function_t function, int32_t index,
                                     iree_string_view_t* key,
                                     iree_string_view_t* value) {
  if (!function.module->get_function_reflection_attr) {
    return IREE_STATUS_NOT_FOUND;
  }
  return function.module->get_function_reflection_attr(
      function.module->self, function.linkage, function.ordinal, index, key,
      value);
}
