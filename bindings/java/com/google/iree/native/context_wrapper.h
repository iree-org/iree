// Copyright 2020 Google LLC
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

#ifndef IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_
#define IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_

#include "bindings/java/com/google/iree/native/function_wrapper.h"
#include "bindings/java/com/google/iree/native/instance_wrapper.h"
#include "bindings/java/com/google/iree/native/module_wrapper.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/context.h"

namespace iree {
namespace java {

class ContextWrapper {
 public:
  Status Create(const InstanceWrapper& instance_wrapper);

  Status CreateWithModules(const InstanceWrapper& instance_wrapper,
                           const std::vector<ModuleWrapper*>& module_wrappers);

  Status RegisterModules(const std::vector<ModuleWrapper*>& module_wrappers);

  Status ResolveFunction(const FunctionWrapper& function_wrapper,
                         iree_string_view_t name);

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

#endif  // IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_CONTEXT_WRAPPER_H_
