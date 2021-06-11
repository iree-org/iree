// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/bindings/java/com/google/iree/native/function_wrapper.h"

namespace iree {
namespace java {

iree_vm_function_t* FunctionWrapper::function() const {
  return function_.get();
}

iree_string_view_t FunctionWrapper::name() const {
  return iree_vm_function_name(function_.get());
}

iree_vm_function_signature_t FunctionWrapper::signature() const {
  return iree_vm_function_signature(function_.get());
}

}  // namespace java
}  // namespace iree
