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

#include "iree/rt/api.h"

#include "absl/time/time.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/api_detail.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/driver_registry.h"
#include "iree/rt/context.h"
#include "iree/rt/debug/debug_server.h"
#include "iree/rt/function.h"
#include "iree/rt/instance.h"
#include "iree/rt/invocation.h"
#include "iree/rt/module.h"
#include "iree/rt/policy.h"

namespace iree {
namespace rt {

//===----------------------------------------------------------------------===//
// iree::rt::Instance
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_instance_create(
    iree_allocator_t allocator, iree_rt_instance_t** out_instance) {
  IREE_TRACE_SCOPE0("iree_rt_instance_create");

  if (!out_instance) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_instance = nullptr;

  auto instance = make_ref<Instance>();
  *out_instance = reinterpret_cast<iree_rt_instance_t*>(instance.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_instance_retain(iree_rt_instance_t* instance) {
  IREE_TRACE_SCOPE0("iree_rt_instance_retain");
  auto* handle = reinterpret_cast<Instance*>(instance);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_instance_release(iree_rt_instance_t* instance) {
  IREE_TRACE_SCOPE0("iree_rt_instance_release");
  auto* handle = reinterpret_cast<Instance*>(instance);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_instance_register_driver_ex(
    iree_rt_instance_t* instance, iree_string_view_t driver_name) {
  IREE_TRACE_SCOPE0("iree_rt_instance_register_driver_ex");
  auto* handle = reinterpret_cast<Instance*>(instance);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto driver, hal::DriverRegistry::shared_registry()->Create(
                       absl::string_view{driver_name.data, driver_name.size}));
  IREE_API_ASSIGN_OR_RETURN(auto available_devices,
                            driver->EnumerateAvailableDevices());
  for (const auto& device_info : available_devices) {
    LOG(INFO) << "  Device: " << device_info.name();
  }
  LOG(INFO) << "Creating default device...";
  IREE_API_ASSIGN_OR_RETURN(auto device, driver->CreateDefaultDevice());
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";
  IREE_API_RETURN_IF_ERROR(
      handle->device_manager()->RegisterDevice(std::move(device)));

  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::rt::Module
//===----------------------------------------------------------------------===//

namespace {

class ExternalModule final : public Module {
 public:
  ExternalModule(iree_rt_external_module_t impl, iree_allocator_t allocator)
      : impl_(impl), allocator_(allocator) {
    IREE_TRACE_SCOPE0("ExternalModule::ctor");
  }

  ~ExternalModule() override {
    IREE_TRACE_SCOPE0("ExternalModule::dtor");
    impl_.destroy(impl_.self);
    std::memset(&impl_, 0, sizeof(impl_));
  }

  absl::string_view name() const override {
    auto result = impl_.name(impl_.self);
    return absl::string_view{result.data, result.size};
  }

  const ModuleSignature signature() const override {
    auto signature = impl_.signature(impl_.self);
    return ModuleSignature{
        signature.import_function_count,
        signature.export_function_count,
        signature.internal_function_count,
        signature.state_slot_count,
    };
  }

  SourceResolver* source_resolver() const override { return nullptr; }

  Disassembler* disassembler() const override { return nullptr; }

  std::string DebugStringShort() const override { return std::string(name()); }

  StatusOr<const Function> LookupFunctionByOrdinal(
      Function::Linkage linkage, int32_t ordinal) const override {
    IREE_TRACE_SCOPE0("ExternalModule::LookupFunctionByOrdinal");
    iree_rt_function_t function;
    auto status = impl_.lookup_function_by_ordinal(
        impl_.self, static_cast<iree_rt_function_linkage_t>(linkage), ordinal,
        &function);
    if (status != IREE_STATUS_OK) {
      return FromApiStatus(status, IREE_LOC);
    }
    return Function{reinterpret_cast<Module*>(function.module),
                    static_cast<Function::Linkage>(function.linkage),
                    function.ordinal};
  }

  StatusOr<const Function> LookupFunctionByName(
      Function::Linkage linkage, absl::string_view name) const override {
    IREE_TRACE_SCOPE0("ExternalModule::LookupFunctionByName");
    iree_rt_function_t function;
    auto status = impl_.lookup_function_by_name(
        impl_.self, static_cast<iree_rt_function_linkage_t>(linkage),
        iree_string_view_t{name.data(), name.size()}, &function);
    if (status != IREE_STATUS_OK) {
      return FromApiStatus(status, IREE_LOC);
    }
    return Function{reinterpret_cast<Module*>(function.module),
                    static_cast<Function::Linkage>(function.linkage),
                    function.ordinal};
  }

  StatusOr<absl::string_view> GetFunctionName(Function::Linkage linkage,
                                              int32_t ordinal) const override {
    IREE_TRACE_SCOPE0("ExternalModule::GetFunctionName");
    iree_string_view_t name;
    auto status = impl_.get_function_name(
        impl_.self, static_cast<iree_rt_function_linkage_t>(linkage), ordinal,
        &name);
    RETURN_IF_ERROR(FromApiStatus(status, IREE_LOC));
    return absl::string_view{name.data, name.size};
  }

  StatusOr<const FunctionSignature> GetFunctionSignature(
      Function::Linkage linkage, int32_t ordinal) const override {
    IREE_TRACE_SCOPE0("ExternalModule::GetFunctionSignature");
    iree_rt_function_signature_t signature;
    auto status = impl_.get_function_signature(
        impl_.self, static_cast<iree_rt_function_linkage_t>(linkage), ordinal,
        &signature);
    if (status != IREE_STATUS_OK) {
      return FromApiStatus(status, IREE_LOC);
    }
    return FunctionSignature{signature.argument_count, signature.result_count};
  }

  Status Execute(
      Stack* stack, const Function function,
      absl::InlinedVector<hal::BufferView, 8> arguments,
      absl::InlinedVector<hal::BufferView, 8>* results) const override {
    // TODO(benvanik): fn ptr callback to external code. Waiting on fibers.
    return UnimplementedErrorBuilder(IREE_LOC)
           << "External calls not yet implemented";
  }

 private:
  iree_rt_external_module_t impl_;
  iree_allocator_t allocator_;
};

}  // namespace

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_module_create_external(
    iree_rt_external_module_t impl, iree_allocator_t allocator,
    iree_rt_module_t** out_module) {
  IREE_TRACE_SCOPE0("iree_rt_module_create_external");

  if (!out_module) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_module = nullptr;

  auto module = make_ref<ExternalModule>(impl, allocator);
  *out_module = reinterpret_cast<iree_rt_module_t*>(module.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_retain(iree_rt_module_t* module) {
  IREE_TRACE_SCOPE0("iree_rt_module_retain");
  auto* handle = reinterpret_cast<Module*>(module);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_release(iree_rt_module_t* module) {
  IREE_TRACE_SCOPE0("iree_rt_module_release");
  auto* handle = reinterpret_cast<Module*>(module);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_rt_module_name(const iree_rt_module_t* module) {
  IREE_TRACE_SCOPE0("iree_rt_module_name");
  const auto* handle = reinterpret_cast<const Module*>(module);
  CHECK(handle) << "NULL module handle";
  return iree_string_view_t{handle->name().data(), handle->name().size()};
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_lookup_function_by_ordinal(iree_rt_module_t* module,
                                          iree_rt_function_linkage_t linkage,
                                          int32_t ordinal,
                                          iree_rt_function_t* out_function) {
  IREE_TRACE_SCOPE0("iree_rt_module_lookup_function_by_ordinal");

  if (!out_function) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_function, 0, sizeof(*out_function));

  auto* handle = reinterpret_cast<Module*>(module);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto function_or = handle->LookupFunctionByOrdinal(
      static_cast<Function::Linkage>(linkage), ordinal);
  if (!function_or.ok()) {
    // Map this invalid argument to not found, per the API spec.
    if (IsInvalidArgument(function_or.status())) {
      return IREE_STATUS_NOT_FOUND;
    }
    return ToApiStatus(std::move(function_or).status());
  }
  auto function = *function_or;

  out_function->module = module;
  out_function->linkage =
      static_cast<iree_rt_function_linkage_t>(function.linkage());
  out_function->ordinal = function.ordinal();

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_module_lookup_function_by_name(iree_rt_module_t* module,
                                       iree_rt_function_linkage_t linkage,
                                       iree_string_view_t name,
                                       iree_rt_function_t* out_function) {
  IREE_TRACE_SCOPE0("iree_rt_module_lookup_function_by_name");

  if (!out_function) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_function, 0, sizeof(*out_function));

  auto* handle = reinterpret_cast<Module*>(module);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto function,
      handle->LookupFunctionByName(static_cast<Function::Linkage>(linkage),
                                   absl::string_view{name.data, name.size}));

  out_function->linkage =
      static_cast<iree_rt_function_linkage_t>(function.linkage());
  out_function->module = module;
  out_function->linkage = linkage;
  out_function->ordinal = function.ordinal();

  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::rt::Function
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_rt_function_name(const iree_rt_function_t* function) {
  IREE_TRACE_SCOPE0("iree_rt_function_name");
  CHECK(function && function->module) << "NULL function handle";
  auto* module = reinterpret_cast<Module*>(function->module);
  auto name_or = module->GetFunctionName(
      static_cast<Function::Linkage>(function->linkage), function->ordinal);
  if (!name_or.ok()) return {};
  auto name = name_or.ValueOrDie();
  return iree_string_view_t{name.data(), name.size()};
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_function_signature(const iree_rt_function_t* function,
                           iree_rt_function_signature_t* out_signature) {
  IREE_TRACE_SCOPE0("iree_rt_function_signature");

  if (!out_signature) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_signature, 0, sizeof(*out_signature));

  if (!function || !function->module) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* module = reinterpret_cast<Module*>(function->module);
  IREE_API_ASSIGN_OR_RETURN(
      auto signature, module->GetFunctionSignature(
                          static_cast<Function::Linkage>(function->linkage),
                          function->ordinal));
  out_signature->argument_count = signature.argument_count();
  out_signature->result_count = signature.result_count();
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::rt::Policy
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_rt_policy_create(
    iree_allocator_t allocator, iree_rt_policy_t** out_policy) {
  IREE_TRACE_SCOPE0("iree_rt_policy_create");

  if (!out_policy) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_policy = nullptr;

  auto policy = make_ref<Policy>();

  *out_policy = reinterpret_cast<iree_rt_policy_t*>(policy.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_policy_retain(iree_rt_policy_t* policy) {
  IREE_TRACE_SCOPE0("iree_rt_policy_retain");
  auto* handle = reinterpret_cast<Policy*>(policy);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_policy_release(iree_rt_policy_t* policy) {
  IREE_TRACE_SCOPE0("iree_rt_policy_release");
  auto* handle = reinterpret_cast<Policy*>(policy);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::rt::Context
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_create(
    iree_rt_instance_t* instance, iree_rt_policy_t* policy,
    iree_allocator_t allocator, iree_rt_context_t** out_context) {
  IREE_TRACE_SCOPE0("iree_rt_context_create");

  if (!out_context) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_context = nullptr;

  if (!instance || !policy) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto context =
      make_ref<Context>(add_ref(reinterpret_cast<Instance*>(instance)),
                        add_ref(reinterpret_cast<Policy*>(policy)));

  *out_context = reinterpret_cast<iree_rt_context_t*>(context.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_retain(iree_rt_context_t* context) {
  IREE_TRACE_SCOPE0("iree_rt_context_retain");
  auto* handle = reinterpret_cast<Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_release(iree_rt_context_t* context) {
  IREE_TRACE_SCOPE0("iree_rt_context_release");
  auto* handle = reinterpret_cast<Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT int32_t IREE_API_CALL
iree_rt_context_id(const iree_rt_context_t* context) {
  IREE_TRACE_SCOPE0("iree_rt_context_id");
  const auto* handle = reinterpret_cast<const Context*>(context);
  CHECK(handle) << "NULL context handle";
  return handle->id();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_register_modules(
    iree_rt_context_t* context, iree_rt_module_t** modules,
    iree_host_size_t module_count) {
  IREE_TRACE_SCOPE0("iree_rt_context_register_modules");
  auto* handle = reinterpret_cast<Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (module_count && !modules) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < module_count; ++i) {
    auto* module = reinterpret_cast<Module*>(modules[i]);
    if (!module) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    IREE_API_RETURN_IF_ERROR(handle->RegisterModule(add_ref(module)));
  }

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_rt_module_t* IREE_API_CALL
iree_rt_context_lookup_module_by_name(const iree_rt_context_t* context,
                                      iree_string_view_t module_name) {
  IREE_TRACE_SCOPE0("iree_rt_context_lookup_module_by_name");
  const auto* handle = reinterpret_cast<const Context*>(context);
  CHECK(handle) << "NULL context handle";
  auto module_or = handle->LookupModuleByName(
      absl::string_view{module_name.data, module_name.size});
  if (!module_or.ok()) {
    return nullptr;
  }
  return reinterpret_cast<iree_rt_module_t*>(module_or.ValueOrDie());
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_context_resolve_function(
    const iree_rt_context_t* context, iree_string_view_t full_name,
    iree_rt_function_t* out_function) {
  IREE_TRACE_SCOPE0("iree_rt_context_resolve_function");

  if (!out_function) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_function, 0, sizeof(*out_function));

  const auto* handle = reinterpret_cast<const Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto full_name_view = absl::string_view{full_name.data, full_name.size};
  size_t last_dot = full_name_view.rfind('.');
  if (last_dot == absl::string_view::npos) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto module_name = full_name_view.substr(0, last_dot);
  auto function_name = full_name_view.substr(last_dot + 1);

  iree_rt_module_t* module = iree_rt_context_lookup_module_by_name(
      context, iree_string_view_t{module_name.data(), module_name.size()});
  if (!module) {
    return IREE_STATUS_NOT_FOUND;
  }

  return iree_rt_module_lookup_function_by_name(
      module, IREE_RT_FUNCTION_LINKAGE_EXPORT,
      iree_string_view_t{function_name.data(), function_name.size()},
      out_function);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_allocate_device_visible_buffer(
    iree_rt_context_t* context, iree_hal_buffer_usage_t buffer_usage,
    iree_host_size_t allocation_size, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_rt_context_allocate_device_visible_buffer");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_buffer, 0, sizeof(*out_buffer));

  const auto* handle = reinterpret_cast<const Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (!allocation_size) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): reroute to context based on current policy.
  auto* device_manager = handle->instance()->device_manager();
  IREE_API_ASSIGN_OR_RETURN(auto device_placement,
                            device_manager->ResolvePlacement({}));
  IREE_API_ASSIGN_OR_RETURN(auto buffer,
                            device_manager->AllocateDeviceVisibleBuffer(
                                static_cast<hal::BufferUsage>(buffer_usage),
                                allocation_size, {device_placement}));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_context_allocate_device_local_buffer(
    iree_rt_context_t* context, iree_hal_buffer_usage_t buffer_usage,
    iree_host_size_t allocation_size, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_rt_context_allocate_device_local_buffer");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_buffer, 0, sizeof(*out_buffer));

  const auto* handle = reinterpret_cast<const Context*>(context);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (!allocation_size) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): reroute to context based on current policy.
  auto* device_manager = handle->instance()->device_manager();
  IREE_API_ASSIGN_OR_RETURN(auto device_placement,
                            device_manager->ResolvePlacement({}));
  IREE_API_ASSIGN_OR_RETURN(auto buffer,
                            device_manager->AllocateDeviceLocalBuffer(
                                static_cast<hal::BufferUsage>(buffer_usage),
                                allocation_size, {device_placement}));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());

  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::rt::Invocation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_create(
    iree_rt_context_t* context, iree_rt_function_t* function,
    iree_rt_policy_t* policy,
    const iree_rt_invocation_dependencies_t* dependencies,
    iree_hal_buffer_view_t** arguments, iree_host_size_t argument_count,
    iree_hal_buffer_view_t** results, iree_host_size_t result_count,
    iree_allocator_t allocator, iree_rt_invocation_t** out_invocation) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_create");

  if (!out_invocation) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_invocation = nullptr;

  if (!context || !function || !function->module) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (dependencies &&
             (dependencies->invocation_count && !dependencies->invocations)) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if ((argument_count && !arguments) || (result_count && !results)) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): unwrap without needing to retain here.
  absl::InlinedVector<ref_ptr<Invocation>, 4> dependent_invocations;
  if (dependencies) {
    dependent_invocations.resize(dependencies->invocation_count);
    for (int i = 0; i < dependencies->invocation_count; ++i) {
      dependent_invocations[i] =
          add_ref(reinterpret_cast<Invocation*>(dependencies->invocations[i]));
    }
  }

  // TODO(benvanik): unwrap without needing to retain here.
  absl::InlinedVector<hal::BufferView, 8> argument_views(argument_count);
  for (int i = 0; i < argument_count; ++i) {
    const auto* api_buffer_view =
        reinterpret_cast<const hal::iree_hal_buffer_view*>(arguments[i]);
    if (!api_buffer_view) {
      return IREE_STATUS_INVALID_ARGUMENT;
    }
    argument_views[i] = hal::BufferView{add_ref(api_buffer_view->impl.buffer),
                                        api_buffer_view->impl.shape,
                                        api_buffer_view->impl.element_size};
  }

  // TODO(benvanik): unwrap without needing to retain here.
  absl::InlinedVector<hal::BufferView, 8> result_views(result_count);
  for (int i = 0; i < result_count; ++i) {
    const auto* api_buffer_view =
        reinterpret_cast<const hal::iree_hal_buffer_view*>(results[i]);
    if (api_buffer_view) {
      result_views[i] = hal::BufferView{add_ref(api_buffer_view->impl.buffer),
                                        api_buffer_view->impl.shape,
                                        api_buffer_view->impl.element_size};
    }
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto invocation,
      Invocation::Create(
          add_ref(reinterpret_cast<Context*>(context)),
          Function{reinterpret_cast<Module*>(function->module),
                   static_cast<Function::Linkage>(function->linkage),
                   function->ordinal},
          add_ref(reinterpret_cast<Policy*>(policy)),
          std::move(dependent_invocations), std::move(argument_views),
          result_views.empty()
              ? absl::optional<absl::InlinedVector<hal::BufferView, 8>>(
                    absl::nullopt)
              : std::move(result_views)));

  *out_invocation =
      reinterpret_cast<iree_rt_invocation_t*>(invocation.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_retain(iree_rt_invocation_t* invocation) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_retain");
  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_release(iree_rt_invocation_t* invocation) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_release");
  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_query_status(iree_rt_invocation_t* invocation) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_query_status");
  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(handle->QueryStatus());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_consume_results(
    iree_rt_invocation_t* invocation, iree_host_size_t result_capacity,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_results,
    iree_host_size_t* out_result_count) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_consume_results");

  if (!out_result_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_result_count = 0;
  if (!out_results) {
    std::memset(out_results, 0,
                sizeof(iree_hal_buffer_view_t*) * result_capacity);
  }

  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  const auto& function = handle->function();
  int32_t result_count = function.signature().result_count();
  *out_result_count = result_count;
  if (!out_results) {
    return IREE_STATUS_OK;
  } else if (result_capacity < result_count) {
    return IREE_STATUS_OUT_OF_RANGE;
  }

  IREE_API_ASSIGN_OR_RETURN(auto results, handle->ConsumeResults());
  iree_status_t status = IREE_STATUS_OK;
  int i = 0;
  for (i = 0; i < results.size(); ++i) {
    iree_shape_t shape;
    status = ToApiShape(results[i].shape, &shape);
    if (status != IREE_STATUS_OK) break;
    status = iree_hal_buffer_view_create(
        reinterpret_cast<iree_hal_buffer_t*>(results[i].buffer.get()), shape,
        results[i].element_size, allocator, &out_results[i]);
    if (status != IREE_STATUS_OK) break;
  }
  if (status != IREE_STATUS_OK) {
    // Release already-retained buffer views on failure.
    for (int j = 0; j < i; ++j) {
      iree_hal_buffer_view_release(out_results[j]);
      out_results[j] = nullptr;
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_rt_invocation_await(
    iree_rt_invocation_t* invocation, iree_time_t deadline) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_await");
  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(handle->Await(ToAbslTime(deadline)));
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_rt_invocation_abort(iree_rt_invocation_t* invocation) {
  IREE_TRACE_SCOPE0("iree_rt_invocation_abort");
  auto* handle = reinterpret_cast<Invocation*>(invocation);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(handle->Abort());
  return IREE_STATUS_OK;
}

}  // namespace rt
}  // namespace iree
