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

#include "experimental/bindings/java/com/google/iree/native/context_wrapper.h"

#include <vector>

#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/vm/ref_cc.h"

namespace iree {
namespace java {

namespace {

std::vector<iree_vm_module_t*> GetModulesFromModuleWrappers(
    const std::vector<ModuleWrapper*>& module_wrappers) {
  std::vector<iree_vm_module_t*> modules(module_wrappers.size());
  for (int i = 0; i < module_wrappers.size(); i++) {
    modules[i] = module_wrappers[i]->module();
  }
  return modules;
}

}  // namespace

Status ContextWrapper::Create(const InstanceWrapper& instance_wrapper) {
  IREE_RETURN_IF_ERROR(iree_vm_context_create(
      instance_wrapper.instance(), iree_allocator_system(), &context_));
  IREE_RETURN_IF_ERROR(CreateDefaultModules());
  std::vector<iree_vm_module_t*> default_modules = {hal_module_};
  IREE_RETURN_IF_ERROR(iree_vm_context_register_modules(
      context_, default_modules.data(), default_modules.size()));
  return OkStatus();
}

Status ContextWrapper::CreateWithModules(
    const InstanceWrapper& instance_wrapper,
    const std::vector<ModuleWrapper*>& module_wrappers) {
  auto modules = GetModulesFromModuleWrappers(module_wrappers);
  IREE_RETURN_IF_ERROR(CreateDefaultModules());

  // The ordering of modules matters, so default modules need to be at the
  // beginning of the vector.
  modules.insert(modules.begin(), hal_module_);

  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance_wrapper.instance(), modules.data(), modules.size(),
      iree_allocator_system(), &context_));
  return OkStatus();
}

Status ContextWrapper::RegisterModules(
    const std::vector<ModuleWrapper*>& module_wrappers) {
  auto modules = GetModulesFromModuleWrappers(module_wrappers);
  IREE_RETURN_IF_ERROR(iree_vm_context_register_modules(
      context_, modules.data(), modules.size()));
  return OkStatus();
}

Status ContextWrapper::ResolveFunction(iree_string_view_t name,
                                       FunctionWrapper* function_wrapper) {
  return iree_vm_context_resolve_function(context_, name,
                                          function_wrapper->function());
}

Status ContextWrapper::InvokeFunction(const FunctionWrapper& function_wrapper,
                                      const std::vector<float*>& inputs,
                                      int input_element_count, float* output) {
  vm::ref<iree_vm_list_t> input_list;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      /*element_type=*/nullptr, input_element_count, iree_allocator_system(),
      &input_list));

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  iree_hal_memory_type_t input_memory_type =
      static_cast<iree_hal_memory_type_t>(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  iree_hal_buffer_usage_t input_buffer_usage =
      static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_ALL |
                                           IREE_HAL_BUFFER_USAGE_CONSTANT);

  for (auto input : inputs) {
    // Write the input into a mappable buffer.
    iree_hal_buffer_t* input_buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        allocator, input_memory_type, input_buffer_usage,
        sizeof(float) * input_element_count, &input_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_write_data(
        input_buffer, 0, input, input_element_count * sizeof(float)));

    // Wrap the input buffers in buffer views.
    iree_hal_buffer_view_t* input_buffer_view = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        input_buffer, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        /*shape=*/&input_element_count,
        /*shape_rank=*/1, , &input_buffer_view));
    iree_hal_buffer_release(input_buffer);

    // Marshal the input buffer views through the input VM variant list.
    auto input_buffer_view_ref =
        iree_hal_buffer_view_move_ref(input_buffer_view);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_ref_move(input_list.get(), &input_buffer_view_ref));
  }

  // Prepare outputs list to accept results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr,
                                           4 * sizeof(float),
                                           iree_allocator_system(), &outputs));

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context_, *function_wrapper.function(),
                                      /*policy=*/nullptr, input_list.get(),
                                      outputs.get(), iree_allocator_system()));

  // Read back the results into the given output buffer.
  auto* output_buffer_view =
      reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
          outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));
  auto* output_buffer = iree_hal_buffer_view_buffer(output_buffer_view);
  // TODO(jennik): this is unsafe - we don't know the size of output ptr here!
  IREE_RETURN_IF_ERROR(iree_hal_buffer_read_data(
      output_buffer, 0, output, iree_hal_buffer_byte_length(output_buffer)));
  return OkStatus();
}

int ContextWrapper::id() const { return iree_vm_context_id(context_); }

ContextWrapper::~ContextWrapper() {
  iree_vm_context_release(context_);
  iree_vm_module_release(hal_module_);
  iree_hal_device_release(device_);
  iree_hal_driver_release(driver_);
}

// TODO(jennik): Also create default string and tensorlist modules.
Status ContextWrapper::CreateDefaultModules() {
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), iree_make_cstring_view("vmla"),
      iree_allocator_system(), &driver_));
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
      driver_, iree_allocator_system(), &device_));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(device_, iree_allocator_system(), &hal_module_));
  return OkStatus();
}

}  // namespace java
}  // namespace iree
