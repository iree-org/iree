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

#include <memory>

#include "experimental/bindings/java/com/google/iree/native/context_wrapper.h"
#include "experimental/bindings/java/com/google/iree/native/function_wrapper.h"
#include "experimental/bindings/java/com/google/iree/native/instance_wrapper.h"
#include "experimental/bindings/java/com/google/iree/native/module_wrapper.h"
#include "experimental/bindings/java/com/google/iree/tests/simple_mul_bytecode_module.h"
#include "iree/base/internal/flags.h"

namespace iree {
namespace java {
namespace {

extern "C" int main(int argc, char** argv) {
  auto instance = std::make_unique<InstanceWrapper>();
  auto instance_status = instance->Create();
  if (!instance_status.ok()) {
    IREE_LOG(ERROR) << "Instance error: " << instance_status.code();
    return 1;
  }
  IREE_LOG(INFO) << "Instance created";

  auto module = std::make_unique<ModuleWrapper>();
  const auto* module_file = simple_mul_bytecode_module_create();
  auto module_status = module->Create(
      reinterpret_cast<const uint8_t*>(module_file->data), module_file->size);
  if (!module_status.ok()) {
    IREE_LOG(ERROR) << "Module error: " << module_status.code();
    return 1;
  }
  std::vector<ModuleWrapper*> modules = {module.get()};
  IREE_LOG(INFO) << "Module created";

  auto context = std::make_unique<ContextWrapper>();
  auto context_status = context->CreateWithModules(*instance, modules);
  if (!context_status.ok()) {
    IREE_LOG(ERROR) << "Context error: " << context_status.code();
    return 1;
  }
  IREE_LOG(INFO) << "Context created";

  FunctionWrapper function;
  const char* function_name = "module.simple_mul";
  auto function_status = context->ResolveFunction(
      iree_string_view_t{function_name, strlen(function_name)}, &function);
  if (!context_status.ok()) {
    IREE_LOG(ERROR) << "Function error: " << function_status.code();
    return 1;
  }
  IREE_LOG(INFO) << "Function created";
  IREE_LOG(INFO) << "Function name: "
                 << std::string(function.name().data, function.name().size);

  float input_x[] = {2.0f, 2.0f, 2.0f, 2.0f};
  float input_y[] = {4.0f, 4.0f, 4.0f, 4.0f};
  std::vector<float*> input{input_x, input_y};
  float output[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  int element_count = 4;

  auto invoke_status =
      context->InvokeFunction(function, input, element_count, output);
  if (!context_status.ok()) {
    IREE_LOG(ERROR) << "Invoke function error: " << function_status.code();
    return 1;
  }

  IREE_LOG(INFO) << "Function output:";
  for (int i = 0; i < element_count; i++) {
    IREE_LOG(INFO) << output[i];
  }

  return 0;
}

}  // namespace
}  // namespace java
}  // namespace iree
