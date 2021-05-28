// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/bindings/java/com/google/iree/native/module_wrapper.h"

namespace iree {
namespace java {

Status ModuleWrapper::Create(const uint8_t* flatbuffer_data,
                             iree_host_size_t length) {
  return iree_vm_bytecode_module_create(
      iree_const_byte_span_t{flatbuffer_data, length}, iree_allocator_null(),
      iree_allocator_system(), &module_);
}

iree_vm_module_t* ModuleWrapper::module() const { return module_; }

iree_string_view_t ModuleWrapper::name() const {
  return iree_vm_module_name(module_);
}

iree_vm_module_signature_t ModuleWrapper::signature() const {
  return iree_vm_module_signature(module_);
}

ModuleWrapper::~ModuleWrapper() { iree_vm_module_release(module_); }

}  // namespace java
}  // namespace iree
