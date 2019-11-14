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

#include "iree/vm2/module_base.h"

#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/tracing.h"

namespace iree {
namespace vm {

absl::string_view Function::name() const {
  iree_string_view_t name;
  if (function_.module->get_function(function_.module->self, function_.linkage,
                                     function_.ordinal, nullptr, &name,
                                     nullptr) != IREE_STATUS_OK) {
    return {};
  }
  return absl::string_view(name.data, name.size);
}

iree_vm_function_signature_t Function::signature() const {
  iree_vm_function_signature_t signature;
  if (function_.module->get_function(function_.module->self, function_.linkage,
                                     function_.ordinal, nullptr, nullptr,
                                     &signature) != IREE_STATUS_OK) {
    return {0};
  }
  return signature;
}

ModuleBase::ModuleBase() {
  IREE_TRACE_SCOPE0("ModuleBase::ctor");

  interface_.self = this;

  interface_.destroy = +[](void* self) {
    IREE_TRACE_SCOPE0("ModuleBase::destroy");
    auto* module = reinterpret_cast<ModuleBase*>(self);
    module->ReleaseReference();
    return IREE_STATUS_OK;
  };

  interface_.name = +[](void* self) {
    auto* module = reinterpret_cast<ModuleBase*>(self);
    auto name = module->name();
    return iree_string_view_t{name.data(), name.size()};
  };

  interface_.signature = +[](void* self) {
    auto* module = reinterpret_cast<ModuleBase*>(self);
    return module->signature();
  };

  interface_.get_function = +[](void* self, iree_vm_function_linkage_t linkage,
                                int32_t ordinal,
                                iree_vm_function_t* out_function,
                                iree_string_view_t* out_name,
                                iree_vm_function_signature_t* out_signature) {
    auto* module = reinterpret_cast<ModuleBase*>(self);
    IREE_API_ASSIGN_OR_RETURN(
        auto function,
        module->GetFunction(static_cast<Function::Linkage>(linkage), ordinal));
    if (out_function) {
      *out_function = function.value();
    }
    if (out_name) {
      auto name = function.name();
      *out_name = iree_string_view_t{name.data(), name.size()};
    }
    if (out_signature) {
      *out_signature = function.signature();
    }
    return IREE_STATUS_OK;
  };

  interface_.lookup_function =
      +[](void* self, iree_vm_function_linkage_t linkage,
          iree_string_view_t name, iree_vm_function_t* out_function) {
        auto* module = reinterpret_cast<ModuleBase*>(self);
        IREE_API_ASSIGN_OR_RETURN(
            auto function,
            module->LookupFunction(static_cast<Function::Linkage>(linkage),
                                   absl::string_view(name.data, name.size)));
        *out_function = function.value();
        return IREE_STATUS_OK;
      };

  interface_.alloc_state = +[](void* self, iree_allocator_t allocator,
                               iree_vm_module_state_t** out_module_state) {
    IREE_TRACE_SCOPE0("ModuleBase::alloc_state");
    auto* module = reinterpret_cast<ModuleBase*>(self);
    IREE_API_ASSIGN_OR_RETURN(auto state, module->CreateState());
    // Good luck, friend.
    *out_module_state =
        reinterpret_cast<iree_vm_module_state_t*>(state.release());
    return IREE_STATUS_OK;
  };

  interface_.free_state =
      +[](void* self, iree_vm_module_state_t* module_state) {
        IREE_TRACE_SCOPE0("ModuleBase::free_state");
        auto* state = reinterpret_cast<State*>(module_state);
        delete state;
        return IREE_STATUS_OK;
      };

  interface_.resolve_import = +[](void* self,
                                  iree_vm_module_state_t* module_state,
                                  int32_t ordinal,
                                  iree_vm_function_t function) {
    auto* state = reinterpret_cast<State*>(module_state);
    IREE_API_RETURN_IF_ERROR(state->ResolveImport(ordinal, Function(function)));
    return IREE_STATUS_OK;
  };

  interface_.execute =
      +[](void* self, iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame,
          iree_vm_execution_result_t* out_result) {
        IREE_TRACE_SCOPE0("ModuleBase::execute");
        auto* module = reinterpret_cast<ModuleBase*>(self);
        IREE_API_ASSIGN_OR_RETURN(auto execution_result,
                                  module->Execute(stack, frame));
        *out_result = execution_result;
        return IREE_STATUS_OK;
      };
}

ModuleBase::~ModuleBase() {
  IREE_TRACE_SCOPE0("ModuleBase::dtor");
  std::memset(&interface_, 0, sizeof(interface_));
}

}  // namespace vm
}  // namespace iree
