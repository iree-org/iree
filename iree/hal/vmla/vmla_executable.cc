// Copyright 2019 Google LLC
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

#include "iree/hal/vmla/vmla_executable.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/host/host_buffer.h"
#include "iree/hal/vmla/vmla_module.h"
#include "iree/schemas/vmla_executable_def_generated.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/invocation.h"
#include "iree/vm/list.h"
#include "iree/vm/module.h"

namespace iree {
namespace hal {
namespace vmla {

// static
StatusOr<ref_ptr<VMLAExecutable>> VMLAExecutable::Load(
    iree_vm_instance_t* instance, iree_vm_module_t* vmla_module,
    ExecutableSpec spec, bool allow_aliasing_data) {
  IREE_TRACE_SCOPE0("VMLAExecutable::Load");
  // Allocate the executable now.
  // We do this here so that if we need to clone the data we are passing that
  // to the VM loader instead of the data we may not have access to later.
  auto executable = make_ref<VMLAExecutable>(spec, allow_aliasing_data);
  IREE_RETURN_IF_ERROR(executable->Initialize(instance, vmla_module));
  return executable;
}

VMLAExecutable::VMLAExecutable(ExecutableSpec spec, bool allow_aliasing_data)
    : spec_(spec) {
  if (!allow_aliasing_data) {
    // Clone data.
    cloned_executable_data_ = {spec.executable_data.begin(),
                               spec.executable_data.end()};
    spec_.executable_data = absl::MakeConstSpan(cloned_executable_data_);
  }
}

VMLAExecutable::~VMLAExecutable() {
  IREE_TRACE_SCOPE0("VMLAExecutable::dtor");
  iree_vm_context_release(context_);
  context_ = nullptr;
}

Status VMLAExecutable::Initialize(iree_vm_instance_t* instance,
                                  iree_vm_module_t* vmla_module) {
  IREE_TRACE_SCOPE0("VMLAExecutable::Initialize");

  if (spec_.executable_data.size() < 16) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Flatbuffer data is not present or less than 16 bytes";
  } else if (!iree::VMLAExecutableDefBufferHasIdentifier(
                 spec_.executable_data.data())) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Flatbuffer data does not have bytecode module identifier";
  }

  const auto* executable_def = ::flatbuffers::GetRoot<iree::VMLAExecutableDef>(
      spec_.executable_data.data());
  if (!executable_def || !executable_def->bytecode_module()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Failed getting root from flatbuffer data";
  }

  // Load bytecode module from the executable spec.
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(
                                 executable_def->bytecode_module()->data()),
                             executable_def->bytecode_module()->size()},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module))
      << "Failed to load executable bytecode module";

  entry_functions_.resize(
      iree_vm_module_signature(bytecode_module).export_function_count);
  for (int i = 0; i < entry_functions_.size(); ++i) {
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i,
        &entry_functions_[i], nullptr));
  }

  // Create context and initialize shared state. Note that each executable here
  // has its own context (and thus its own vmla.interface instance).
  std::array<iree_vm_module_t*, 2> modules = {vmla_module, bytecode_module};
  auto result = StatusBuilder(iree_vm_context_create_with_modules(
                                  instance, modules.data(), modules.size(),
                                  iree_allocator_system(), &context_),
                              IREE_LOC)
                << "Failed resolving imports for executable module";
  iree_vm_module_release(bytecode_module);

  return std::move(result);
}

struct VMLADispatchState : public HostExecutable::DispatchState {
  VMLADispatchState() { interface_ref = Interface_retain_ref(&interface); }
  ~VMLADispatchState() override { iree_vm_ref_release(&interface_ref); }

  iree_vm_function_t function;
  Interface interface;
  iree_vm_ref_t interface_ref;
  iree_host_size_t input_list_size = 0;
};

StatusOr<ref_ptr<HostExecutable::DispatchState>>
VMLAExecutable::PrepareDispatch(const DispatchParams& params) {
  IREE_TRACE_SCOPE0("VMLAExecutable::PrepareDispatch");

  if (params.entry_point >= entry_functions_.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Invalid entry point ordinal " << params.entry_point;
  }

  auto dispatch_state = make_ref<VMLADispatchState>();
  dispatch_state->function = entry_functions_[params.entry_point];
  dispatch_state->input_list_size = iree_vm_list_storage_size(
      /*element_type=*/nullptr, /*interface*/ 1 + /*workgroup_xyz[3]*/ 3);

  auto* interface = &dispatch_state->interface;
  IREE_RETURN_IF_ERROR(interface->SetConstants(params.push_constants->values));

  for (int set_ordinal = 0; set_ordinal < params.set_bindings.size();
       ++set_ordinal) {
    for (const auto& binding : params.set_bindings[set_ordinal]) {
      // TODO(benvanik): plumb binding directly into VMLA to avoid this.
      void* data = static_cast<HostBuffer*>(binding.buffer->allocated_buffer())
                       ->mutable_data();
      data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(data) +
                                     binding.buffer->byte_offset());
      IREE_ASSIGN_OR_RETURN(
          auto buffer, Buffer::WrapMutable(data, binding.buffer->byte_length(),
                                           iree_allocator_null()));
      IREE_RETURN_IF_ERROR(interface->SetBinding(set_ordinal, binding.binding,
                                                 {std::move(buffer)}));
    }
  }

  return std::move(dispatch_state);
}

Status VMLAExecutable::DispatchTile(DispatchState* state,
                                    std::array<uint32_t, 3> workgroup_xyz) {
  IREE_TRACE_SCOPE0("VMLAExecutable::DispatchTile");
  auto* dispatch_state = static_cast<VMLADispatchState*>(state);

  auto* input_list_storage = alloca(dispatch_state->input_list_size);
  iree_vm_list_t* input_list = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_list_initialize(
      iree_make_byte_span(input_list_storage, dispatch_state->input_list_size),
      /*element_type=*/nullptr,
      /*interface*/ 1 + /*workgroup_xyz[3]*/ 3, &input_list));
  iree_vm_list_push_ref_retain(input_list, &dispatch_state->interface_ref);
  for (int i = 0; i < workgroup_xyz.size(); ++i) {
    iree_vm_value_t value = iree_vm_value_make_i32(workgroup_xyz[i]);
    iree_vm_list_push_value(input_list, &value);
  }

  auto status =
      Status(iree_vm_invoke(context(), dispatch_state->function,
                            /*policy=*/nullptr, input_list,
                            /*outputs=*/nullptr, iree_allocator_system()));

  iree_vm_list_deinitialize(input_list);

  return status;
}

}  // namespace vmla
}  // namespace hal
}  // namespace iree
