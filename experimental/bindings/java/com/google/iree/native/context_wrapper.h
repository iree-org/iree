// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_
#define IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_

#include <vector>

#include "experimental/bindings/java/com/google/iree/native/function_wrapper.h"
#include "experimental/bindings/java/com/google/iree/native/instance_wrapper.h"
#include "experimental/bindings/java/com/google/iree/native/module_wrapper.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/api.h"

namespace iree {
namespace java {

class ContextWrapper {
 public:
  Status Create(const InstanceWrapper& instance_wrapper);

  Status CreateWithModules(const InstanceWrapper& instance_wrapper,
                           const std::vector<ModuleWrapper*>& module_wrappers);

  Status RegisterModules(const std::vector<ModuleWrapper*>& module_wrappers);

  Status ResolveFunction(iree_string_view_t name,
                         FunctionWrapper* function_wrapper);

  // TODO(jennik): Support other input types aside from floats.
  Status InvokeFunction(const FunctionWrapper& function_wrapper,
                        const std::vector<float*>& inputs,
                        int input_element_count, float* output);

  int id() const;

  ~ContextWrapper();

 private:
  Status CreateDefaultModules();

  iree_vm_context_t* context_ = nullptr;
  // TODO(jennik): These need to be configurable on the java side.
  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_vm_module_t* hal_module_ = nullptr;
};

}  // namespace java
}  // namespace iree

#endif  // IREE_EXPERIMENTAL_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_
