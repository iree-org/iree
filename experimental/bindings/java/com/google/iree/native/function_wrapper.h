// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_FUNCTION_WRAPPER_H_
#define IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_FUNCTION_WRAPPER_H_

#include <memory>

#include "iree/vm/api.h"

namespace iree {
namespace java {

class FunctionWrapper {
 public:
  iree_vm_function_t* function() const;

  iree_string_view_t name() const;

  iree_vm_function_signature_t signature() const;

 private:
  std::unique_ptr<iree_vm_function_t> function_ =
      std::make_unique<iree_vm_function_t>();
};

}  // namespace java
}  // namespace iree

#endif  // IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_FUNCTION_WRAPPER_H_
