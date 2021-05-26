// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_
#define IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_

#include "iree/base/status.h"
#include "iree/vm/bytecode_module.h"

namespace iree {
namespace java {

class ModuleWrapper {
 public:
  Status Create(const uint8_t* flatbuffer_data, iree_host_size_t length);

  iree_vm_module_t* module() const;

  iree_string_view_t name() const;

  iree_vm_module_signature_t signature() const;

  ~ModuleWrapper();

 private:
  iree_vm_module_t* module_ = nullptr;
};

}  // namespace java
}  // namespace iree

#endif  // IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_
