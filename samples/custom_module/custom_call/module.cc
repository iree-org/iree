// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module.h"

#include <cstring>
#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "iree/base/internal/span.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

namespace {

using namespace iree;

StatusOr<int> FormatToSize(const char* format, iree_host_size_t formatSize) {
  int outSize = 0;
  for (auto i = 0; i < formatSize; i++) {
    switch (format[i]) {
      case IREE_VM_CCONV_TYPE_I32:
        outSize += sizeof(int);
        continue;
      case IREE_VM_CCONV_TYPE_I64:
        outSize += sizeof(int64_t);
        continue;
      case IREE_VM_CCONV_TYPE_F32:
        outSize += sizeof(float);
        continue;
      case IREE_VM_CCONV_TYPE_F64:
        outSize += sizeof(double);
        continue;
      case IREE_VM_CCONV_TYPE_REF:
        outSize += sizeof(uint8_t*);
        continue;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "Format %c unimplemented",
                                format[i]);
    }
  }
  return outSize;
}

StatusOr<iree_vm_ref_t*> GetBufferViewRefPtr(uint8_t* ptr) {
  auto* refPtr = reinterpret_cast<iree_vm_ref_t*>(ptr);
  if (refPtr->type !=
      vm::ref_type_descriptor<iree_hal_buffer_view_t>::type()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter contains a reference to the wrong type; "
        "have %.*s but expected iree_hal_buffer_view_t",
        (int)iree_vm_ref_type_name(refPtr->type).size,
        iree_vm_ref_type_name(refPtr->type).data);
  }
  return refPtr;
}

size_t GetFormatSize(const char c) {
  switch (c) {
    case IREE_VM_CCONV_TYPE_I32:
      return sizeof(int);
    case IREE_VM_CCONV_TYPE_I64:
      return sizeof(int64_t);
    case IREE_VM_CCONV_TYPE_F32:
      return sizeof(float);
    case IREE_VM_CCONV_TYPE_F64:
      return sizeof(double);
    default:
      return 0;
  }
}

StatusOr<uint8_t*> FormattedToRaw(
    const char* format, iree_host_size_t formatSize, uint8_t* formattedPtr,
    uint8_t* raw) {
  for (auto i = 0; i < formatSize; i++) {
    switch (format[i]) {
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F32:
      case IREE_VM_CCONV_TYPE_F64: {
          auto size = GetFormatSize(format[i]);
          memcpy((void*)raw, (void*)formattedPtr, size);
          formattedPtr += size;
          raw += size;
        }
        continue;
      case IREE_VM_CCONV_TYPE_REF: {
          IREE_ASSIGN_OR_RETURN(iree_vm_ref_t* refPtr,
                                GetBufferViewRefPtr(formattedPtr));
          iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(
              (const iree_hal_buffer_view_t*)refPtr->ptr);
          iree_hal_buffer_mapping_t mapping = {{0}};
          Status status = iree_hal_buffer_map_range(buffer,
            IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_ANY, 0,
            IREE_WHOLE_BUFFER, &mapping);
          if (!status.ok())
            return status;
          memcpy((void*)raw, (void*)&mapping.contents.data, sizeof(uint8_t*));
          iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
          formattedPtr += sizeof(iree_vm_ref_t);
          raw += sizeof(uint8_t*);
        }
        continue;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "Format %c unimplemented",
                                format[i]);
    }
  }
  return formattedPtr;
}

Status CopyBetweenFormatted(const char* format, iree_host_size_t formatSize,
                            uint8_t* inFormattedPtr, uint8_t* outFormattedPtr) {
  for (auto i = 0; i < formatSize; i++) {
    switch (format[i]) {
      case IREE_VM_CCONV_TYPE_I32:
      case IREE_VM_CCONV_TYPE_I64:
      case IREE_VM_CCONV_TYPE_F32:
      case IREE_VM_CCONV_TYPE_F64: {
          auto size = GetFormatSize(format[i]);
          memcpy((void*)outFormattedPtr, (void*)inFormattedPtr, size);
          inFormattedPtr += size;
          outFormattedPtr += size;
        }
        continue;
      case IREE_VM_CCONV_TYPE_REF: {
          IREE_ASSIGN_OR_RETURN(iree_vm_ref_t* refPtr,
                                GetBufferViewRefPtr(inFormattedPtr));
          iree_vm_ref_move(refPtr,
                           reinterpret_cast<iree_vm_ref_t*>(outFormattedPtr));
        }
        continue;
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "Format %c unimplemented",
                                format[i]);
    }
  }
  return OkStatus();
}

Status GetFenceRefPtr(uint8_t* ptr, vm::ref<iree_hal_fence_t>& fence) {
  auto* refPtr = reinterpret_cast<iree_vm_ref_t*>(ptr);
  if (refPtr->type !=
      vm::ref_type_descriptor<iree_hal_fence_t>::type()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter contains a reference to the wrong type; "
        "have %.*s but expected iree_hal_fence_t",
        (int)iree_vm_ref_type_name(refPtr->type).size,
        iree_vm_ref_type_name(refPtr->type).data);
  }
  fence = vm::retain_ref(reinterpret_cast<iree_hal_fence_t*>(refPtr->ptr));
  memset(refPtr, 0, sizeof(*refPtr));
  return OkStatus();
}

Status GetFences(
    const char* format, uint8_t* formattedPtr,
    vm::ref<iree_hal_fence_t>& waitFence,
    vm::ref<iree_hal_fence_t>& signalFence) {
  if (!(format[0] == IREE_VM_CCONV_TYPE_REF &&
        format[1] == IREE_VM_CCONV_TYPE_REF)) {
     return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "Format %c, %c unexpected; "
                             "fence references expected",
                             format[0], format[1]);
  }
  Status status = GetFenceRefPtr(formattedPtr, waitFence);
  if (!status.ok())
    return status;
  status = GetFenceRefPtr(formattedPtr + sizeof(iree_vm_ref_t), signalFence);
  return status;
}

Status CustomCall(CustomCallPtr fnPtr, iree_vm_function_call_t call) {
  iree_string_view_t functionName = iree_vm_function_name(&call.function);
  printf("CustomCall name: %.*s\n", (int)functionName.size, functionName.data);
  auto argResFormatSize = call.format_arguments.size - 2;
  auto argResFormatData = call.format_arguments.data;
  auto resFormatSize = call.format_results.size;
  auto resFormatData = call.format_results.data;
  IREE_ASSIGN_OR_RETURN(
      int argResSize, FormatToSize(argResFormatData, argResFormatSize));
  std::unique_ptr<uint8_t[]> argResList(new uint8_t[argResSize]());
  IREE_ASSIGN_OR_RETURN(
      uint8_t* fencePtr,
      FormattedToRaw(argResFormatData, argResFormatSize,
                     call.arguments.data, argResList.get()));
  vm::ref<iree_hal_fence_t> waitFence;
  vm::ref<iree_hal_fence_t> signalFence;
  Status status = GetFences(argResFormatData + argResFormatSize, fencePtr,
                            waitFence, signalFence);
  if (!status.ok())
    return status;

  std::optional<std::string> error;
  IREE_ASSIGN_OR_RETURN(
    int resSize, FormatToSize(resFormatData, resFormatSize));
  uint8_t* resList = argResList.get();
  uint8_t* argList = argResList.get() + resSize;

  status = iree_hal_fence_wait(waitFence.get(), iree_infinite_timeout());
  if (!status.ok())
    return status;

  fnPtr((void*)resList, (void **)argList);

  status = CopyBetweenFormatted(resFormatData, resFormatSize,
                                call.arguments.data, call.results.data);
  if (!status.ok())
    return status;
  if(error.has_value()) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "Custom Call returned an error");
  }
  status = iree_hal_fence_signal(signalFence.get());
  if (!status.ok())
    iree_hal_fence_fail(signalFence.get(), status.release());

  return OkStatus();
}

class CustomCallModule final {
  using State = std::monostate;
 public:
  CustomCallModule(const char* name, uint32_t version, iree_vm_instance_t* instance,
                   iree_allocator_t allocator,
                   iree::span<std::pair<iree_string_view_t,
                                        CustomCallPtr>> dispatch_table)
      : name_(name),
        version_(version),
        instance_(instance),
        allocator_(allocator),
        dispatch_table_(dispatch_table) {
    iree_vm_instance_retain(instance);
    IREE_CHECK_OK(iree_vm_module_initialize(&interface_, this));
    interface_.destroy = CustomCallModule::ModuleDestroy;
    interface_.name = CustomCallModule::ModuleName;
    interface_.signature = CustomCallModule::ModuleSignature;
    // TODO(benvanik): get_module_attr
    interface_.enumerate_dependencies =
        CustomCallModule::ModuleEnumerateDependencies;
    interface_.lookup_function = CustomCallModule::ModuleLookupFunction;
    interface_.get_function = CustomCallModule::ModuleGetFunction;
    // TODO(benvanik): get_function_attr
    interface_.alloc_state = CustomCallModule::ModuleAllocState;
    interface_.free_state = CustomCallModule::ModuleFreeState;
    interface_.resolve_import = CustomCallModule::ModuleResolveImport;
    interface_.notify = CustomCallModule::ModuleNotify;
    interface_.begin_call = CustomCallModule::ModuleBeginCall;
    // TODO(benvanik): resume_call
  }

  ~CustomCallModule() { iree_vm_instance_release(instance_); }

  iree_vm_instance_t* instance() const { return instance_; }

  // C API module interface bound to this CustomCallModule instance.
  iree_vm_module_t* interface() { return &interface_; }

 private:
  // Enumerates module dependencies by issuing |callback| for each dependency.
  Status EnumerateDependencies(
      std::function<Status(const iree_vm_module_dependency_t*)> callback) {
    return OkStatus();
  }

  // Notifies the module a signal has been raised.
  Status Notify(State* state, iree_vm_signal_t signal) {
    return OkStatus();
  }

  static CustomCallModule* FromModulePointer(void* self) {
    return reinterpret_cast<CustomCallModule*>(self);
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
      *out_name = dispatch_function.first;
    }
    if (out_signature) {
      out_signature->calling_convention = iree_string_view_empty();
    }
    return iree_ok_status();
  }

  static iree_status_t ModuleLookupFunction(void* self,
                                            iree_vm_function_linkage_t linkage,
                                            iree_string_view_t name,
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
      if (iree_string_view_equal(name, module->dispatch_table_[i].first)) {
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

    auto module_state = std::make_unique<State>();

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

    iree_status_t status = CustomCall(info.second, call);

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      status = iree_status_annotate_f(
          status, "while invoking C++ function %s.%.*s", module->name_,
          (int)info.first.size, info.first.data);
      return status;
    }

    return iree_vm_stack_function_leave(stack);
  }

  const char* name_;
  uint32_t version_;
  iree_vm_instance_t* instance_;
  const iree_allocator_t allocator_;
  iree_vm_module_t interface_;

  const iree::span<std::pair<iree_string_view_t,
                             CustomCallPtr>> dispatch_table_;
};

}  // namespace

std::vector<std::pair<iree_string_view_t, CustomCallPtr>> customCallFunctions;

extern "C" void AddCustomCall(const char* name, CustomCallPtr fnPtr) {
  customCallFunctions.push_back(std::make_pair(iree_make_cstring_view(name),
                                               fnPtr));
}

// Note that while we are using C++ bindings internally we still expose the
// module as a C instance. This hides the details of our implementation.
extern "C" void iree_custom_module_custom_call_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_runtime_session_t* session) {
  iree_vm_module_t* custom_module = NULL;
  auto module = std::make_unique<CustomCallModule>(
      "custom_call", /*version=*/0, instance, host_allocator,
      iree::span<std::pair<iree_string_view_t, CustomCallPtr>>(
          customCallFunctions));
  custom_module = module.release()->interface();
  IREE_CHECK_OK(iree_runtime_session_append_module(session, custom_module));
  iree_vm_module_release(custom_module);
}
