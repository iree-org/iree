// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_NATIVE_MODULE_CC_H_
#define IREE_VM_NATIVE_MODULE_CC_H_

#include <cstring>
#include <functional>
#include <memory>

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/native_module_packing.h"  // IWYU pragma: export
#include "iree/vm/stack.h"

#ifndef __cplusplus
#error "This header is meant for use with C++ module implementations."
#endif  // __cplusplus

namespace iree {
namespace vm {

// A native module as exported to the VM dynamic module linking API.
// This allows easy wrapping of C++ module implementations and removes a
// majority of the boilerplate required with marshaling args/results out/in of
// the VM via the ABI.
//
// Functions are defined on the State type as member functions returning either
// Status or StatusOr. Arguments are passed as primitive types (int32_t),
// wrapped ref objects (vm::ref<my_type_t>&), or some nesting of std::array,
// std::tuple, and std::span to match fixed-length arrays of the same type,
// tuples of mixed types, or dynamic arrays (variadic arguments). Results may be
// returned as either their type or an std::tuple/std::array of types.
//
// Usage:
//   // Per-context module state that must only be thread-compatible.
//   // Define
//   struct MyState final {
//     StatusOr<std::tuple<int32_t, int32_t>> MyMethod1(vm::ref<my_type_t> t);
//   };
//
//   // Table of functions mapped to their name in the IR.
//   static const vm::NativeFunction<MyState> kMyFunctions[] = {
//     vm::MakeNativeFunction("my_method_1", &MyState::MyMethod1),
//   };
//
//   // The outer module wrapper shared across contexts.
//   // Must be thread-safe.
//   struct MyModule : public NativeModule<MyState> {
//     StatusOr<std::unique_ptr<MyState>> CreateState(iree_allocator_t) {
//       // You could pass in thread-safe shared resources to MyState.
//       return std::make_unique<MyState>();
//     }
//   };
//
//   // Creates the module and exposes it as a C interface.
//   // Ownership transfers to the caller.
//   iree_vm_module_t* create_my_module(iree_allocator_t allocator) {
//     return std::make_unique<MyModule>("my_module", allocator,
//         std::span{kCustomModuleFunctions}).release()->interface();
//   }
template <typename State>
class NativeModule {
 public:
  NativeModule(const char* name, uint32_t version, iree_vm_instance_t* instance,
               iree_allocator_t allocator,
               iree::span<const NativeFunction<State>> dispatch_table)
      : name_(name),
        version_(version),
        instance_(instance),
        allocator_(allocator),
        dispatch_table_(dispatch_table) {
    iree_vm_instance_retain(instance);
    IREE_CHECK_OK(iree_vm_module_initialize(&interface_, this));
    interface_.destroy = NativeModule::ModuleDestroy;
    interface_.name = NativeModule::ModuleName;
    interface_.signature = NativeModule::ModuleSignature;
    // TODO(benvanik): get_module_attr
    interface_.enumerate_dependencies =
        NativeModule::ModuleEnumerateDependencies;
    interface_.lookup_function = NativeModule::ModuleLookupFunction;
    interface_.get_function = NativeModule::ModuleGetFunction;
    // TODO(benvanik): get_function_attr
    interface_.alloc_state = NativeModule::ModuleAllocState;
    interface_.free_state = NativeModule::ModuleFreeState;
    interface_.resolve_import = NativeModule::ModuleResolveImport;
    interface_.notify = NativeModule::ModuleNotify;
    interface_.begin_call = NativeModule::ModuleBeginCall;
    // TODO(benvanik): resume_call
  }

  virtual ~NativeModule() { iree_vm_instance_release(instance_); }

  iree_vm_instance_t* instance() const { return instance_; }

  // C API module interface bound to this NativeModule instance.
  iree_vm_module_t* interface() { return &interface_; }

 protected:
  // Enumerates module dependencies by issuing |callback| for each dependency.
  virtual Status EnumerateDependencies(
      std::function<Status(const iree_vm_module_dependency_t*)> callback) {
    return OkStatus();
  }

  // Creates a new per-context module State holder.
  virtual StatusOr<std::unique_ptr<State>> CreateState(
      iree_allocator_t allocator) = 0;

  // Notifies the module a signal has been raised.
  virtual Status Notify(State* state, iree_vm_signal_t signal) {
    return OkStatus();
  }

 private:
  static NativeModule* FromModulePointer(void* self) {
    return reinterpret_cast<NativeModule*>(self);
  }
  static State* FromStatePointer(void* self) {
    return reinterpret_cast<State*>(self);
  }

  static void ModuleDestroy(void* self) { delete FromModulePointer(self); }

  static iree_string_view_t ModuleName(void* self) {
    auto* module = FromModulePointer(self);
    return iree_make_cstring_view(module->name_);
  }

  static iree_vm_module_signature_t ModuleSignature(void* self) {
    auto* module = FromModulePointer(self);
    iree_vm_module_signature_t signature = {0};
    signature.version = module->version_;
    signature.attr_count = 0;
    signature.import_function_count = 0;
    signature.export_function_count = module->dispatch_table_.size();
    signature.internal_function_count = 0;
    return signature;
  }

  static iree_status_t ModuleEnumerateDependencies(
      void* self, iree_vm_module_dependency_callback_t callback,
      void* user_data) {
    auto* module = FromModulePointer(self);
    auto callback_fn =
        [callback, user_data](const iree_vm_module_dependency_t* dependency) {
          return Status(callback(user_data, dependency));
        };
    IREE_RETURN_IF_ERROR(module->EnumerateDependencies(std::move(callback_fn)));
    return OkStatus();
  }

  static iree_status_t ModuleGetFunction(
      void* self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_vm_function_t* out_function, iree_string_view_t* out_name,
      iree_vm_function_signature_t* out_signature) {
    if (out_function) {
      std::memset(out_function, 0, sizeof(*out_function));
    }
    if (out_name) {
      out_name->data = nullptr;
      out_name->size = 0;
    }
    if (out_signature) {
      std::memset(out_signature, 0, sizeof(*out_signature));
    }
    auto* module = FromModulePointer(self);
    if (IREE_UNLIKELY(ordinal > module->dispatch_table_.size())) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function out of bounds: 0 < %zu < %zu", ordinal,
                              module->dispatch_table_.size());
    }
    const auto& dispatch_function = module->dispatch_table_[ordinal];
    if (out_function) {
      out_function->module = module->interface();
      out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
      out_function->ordinal = static_cast<uint16_t>(ordinal);
    }
    if (out_name) {
      *out_name = dispatch_function.name;
    }
    if (out_signature) {
      out_signature->calling_convention = dispatch_function.cconv;
    }
    return iree_ok_status();
  }

  static iree_status_t ModuleLookupFunction(
      void* self, iree_vm_function_linkage_t linkage, iree_string_view_t name,
      const iree_vm_function_signature_t* expected_signature,
      iree_vm_function_t* out_function) {
    IREE_ASSERT_ARGUMENT(out_function);
    std::memset(out_function, 0, sizeof(*out_function));
    if (IREE_UNLIKELY(!name.data || !name.size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function name empty");
    }

    auto* module = FromModulePointer(self);
    out_function->module = module->interface();
    out_function->linkage = IREE_VM_FUNCTION_LINKAGE_EXPORT;
    for (int i = 0; i < module->dispatch_table_.size(); ++i) {
      if (iree_string_view_equal(name, module->dispatch_table_[i].name)) {
        out_function->ordinal = i;
        return iree_ok_status();
      }
    }

    return iree_make_status(IREE_STATUS_NOT_FOUND, "function %.*s not exported",
                            (int)name.size, name.data);
  }

  static iree_status_t ModuleAllocState(
      void* self, iree_allocator_t allocator,
      iree_vm_module_state_t** out_module_state) {
    IREE_ASSERT_ARGUMENT(out_module_state);
    *out_module_state = nullptr;

    auto* module = FromModulePointer(self);
    IREE_ASSIGN_OR_RETURN(auto module_state, module->CreateState(allocator));

    *out_module_state =
        reinterpret_cast<iree_vm_module_state_t*>(module_state.release());
    return iree_ok_status();
  }

  static void ModuleFreeState(void* self,
                              iree_vm_module_state_t* module_state) {
    if (module_state) delete FromStatePointer(module_state);
  }

  static iree_status_t ModuleResolveImport(
      void* self, iree_vm_module_state_t* module_state,
      iree_host_size_t ordinal, const iree_vm_function_t* function,
      const iree_vm_function_signature_t* signature) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "C++ API does not support imports");
  }

  static iree_status_t ModuleNotify(void* self,
                                    iree_vm_module_state_t* module_state,
                                    iree_vm_signal_t signal) {
    auto* module = FromModulePointer(self);
    return module->Notify(FromStatePointer(module_state), signal);
  }

  static iree_status_t ModuleBeginCall(void* self, iree_vm_stack_t* stack,
                                       iree_vm_function_call_t call) {
    auto* module = FromModulePointer(self);
    if (IREE_UNLIKELY(call.function.ordinal >=
                      module->dispatch_table_.size())) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function ordinal out of bounds: 0 < %u < %zu",
                              call.function.ordinal,
                              module->dispatch_table_.size());
    }
    const auto& info = module->dispatch_table_[call.function.ordinal];

    // NOTE: VM stack is currently unused. We could stash things here for the
    // debugger or use it for coroutine state.
    iree_host_size_t frame_size = 0;

    iree_vm_stack_frame_t* callee_frame = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_stack_function_enter(
        stack, &call.function, IREE_VM_STACK_FRAME_NATIVE, frame_size,
        /*frame_cleanup_fn=*/nullptr, &callee_frame));

    auto* state = FromStatePointer(callee_frame->module_state);
    iree_status_t status = info.call(info.ptr, state, stack, call);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      status = iree_status_annotate_f(
          status, "while invoking C++ function %s.%.*s", module->name_,
          (int)info.name.size, info.name.data);
      return status;
    }

    return iree_vm_stack_function_leave(stack);
  }

  const char* name_;
  uint32_t version_;
  iree_vm_instance_t* instance_;
  const iree_allocator_t allocator_;
  iree_vm_module_t interface_;

  const iree::span<const NativeFunction<State>> dispatch_table_;
};

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_NATIVE_MODULE_CC_H_
