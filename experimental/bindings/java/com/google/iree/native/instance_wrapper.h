// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_INSTANCE_WRAPPER_H_
#define IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_INSTANCE_WRAPPER_H_

#include "iree/base/status.h"
#include "iree/vm/api.h"

namespace iree {
namespace java {

class InstanceWrapper {
 public:
  Status Create();

  iree_vm_instance_t* instance() const;

  ~InstanceWrapper();

 private:
  iree_vm_instance_t* instance_ = nullptr;
};

}  // namespace java
}  // namespace iree

#endif  // IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_INSTANCE_WRAPPER_H_
