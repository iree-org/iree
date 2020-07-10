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

#ifndef IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_
#define IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_

#include "iree/base/status.h"
#include "iree/vm/bytecode_module.h"

namespace iree {
namespace java {

class ModuleWrapper {
 public:
  Status Create(const uint8_t* flatbuffer_data, iree_host_size_t length);

  ~ModuleWrapper();

 private:
  iree_vm_module_t* module_ = nullptr;
};

}  // namespace java
}  // namespace iree

#endif  // IREE_BINDINGS_JAVA_COM_GOOGLE_IREE_NATIVE_MODULE_WRAPPER_H_
