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

#include "iree/vm/builtin_types.h"

static iree_vm_ref_type_descriptor_t iree_vm_ro_byte_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_vm_rw_byte_buffer_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_ro_byte_buffer, iree_vm_ro_byte_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_rw_byte_buffer, iree_vm_rw_byte_buffer_t);

static void iree_vm_ro_byte_buffer_destroy(void* ptr) {
  iree_vm_ro_byte_buffer_t* ref = (iree_vm_ro_byte_buffer_t*)ptr;
  if (ref->destroy) {
    ref->destroy(ptr);
  }
}

static void iree_vm_rw_byte_buffer_destroy(void* ptr) {
  iree_vm_rw_byte_buffer_t* ref = (iree_vm_rw_byte_buffer_t*)ptr;
  if (ref->destroy) {
    ref->destroy(ptr);
  }
}

iree_status_t iree_vm_list_register_types();

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_vm_register_builtin_types() {
  if (iree_vm_ro_byte_buffer_descriptor.type != IREE_VM_REF_TYPE_NULL) {
    return iree_ok_status();
  }

  iree_vm_ro_byte_buffer_descriptor.destroy = iree_vm_ro_byte_buffer_destroy;
  iree_vm_ro_byte_buffer_descriptor.offsetof_counter =
      offsetof(iree_vm_ro_byte_buffer_t, ref_object.counter);
  iree_vm_ro_byte_buffer_descriptor.type_name =
      iree_make_cstring_view("iree.byte_buffer");
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_register_type(&iree_vm_ro_byte_buffer_descriptor));

  iree_vm_rw_byte_buffer_descriptor.destroy = iree_vm_rw_byte_buffer_destroy;
  iree_vm_rw_byte_buffer_descriptor.offsetof_counter =
      offsetof(iree_vm_rw_byte_buffer_t, ref_object.counter);
  iree_vm_rw_byte_buffer_descriptor.type_name =
      iree_make_cstring_view("iree.mutable_byte_buffer");
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_register_type(&iree_vm_rw_byte_buffer_descriptor));

  IREE_RETURN_IF_ERROR(iree_vm_list_register_types());

  return iree_ok_status();
}
