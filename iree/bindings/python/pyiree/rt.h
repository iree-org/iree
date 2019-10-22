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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_RT_H_
#define IREE_BINDINGS_PYTHON_PYIREE_RT_H_

#include "absl/container/inlined_vector.h"
#include "iree/bindings/python/pyiree/binding.h"
#include "iree/bindings/python/pyiree/hal.h"
#include "iree/bindings/python/pyiree/status_utils.h"
#include "iree/rt/api.h"

namespace iree {
namespace python {

// Adapts API pointer access to retain/release API calls.
template <>
struct ApiPtrAdapter<iree_rt_module_t> {
  static void Retain(iree_rt_module_t* m) { iree_rt_module_retain(m); }
  static void Release(iree_rt_module_t* m) { iree_rt_module_release(m); }
};

template <>
struct ApiPtrAdapter<iree_rt_instance_t> {
  static void Retain(iree_rt_instance_t* inst) {
    iree_rt_instance_retain(inst);
  }
  static void Release(iree_rt_instance_t* inst) {
    iree_rt_instance_release(inst);
  }
};

template <>
struct ApiPtrAdapter<iree_rt_policy_t> {
  static void Retain(iree_rt_policy_t* p) { iree_rt_policy_retain(p); }
  static void Release(iree_rt_policy_t* p) { iree_rt_policy_release(p); }
};

template <>
struct ApiPtrAdapter<iree_rt_context_t> {
  static void Retain(iree_rt_context_t* c) { iree_rt_context_retain(c); }
  static void Release(iree_rt_context_t* c) { iree_rt_context_release(c); }
};

template <>
struct ApiPtrAdapter<iree_rt_invocation_t> {
  static void Retain(iree_rt_invocation_t* inv) {
    iree_rt_invocation_retain(inv);
  }
  static void Release(iree_rt_invocation_t* inv) {
    iree_rt_invocation_release(inv);
  }
};

// Wrapper classes. These mirror the Python declarations.
class RtFunction {
 public:
  // Note that this will retain the module.
  RtFunction(iree_rt_function_t function) : function_(function) {
    iree_rt_module_retain(function_.module);
  }
  ~RtFunction() {
    if (function_.module) iree_rt_module_release(function_.module);
  }
  RtFunction(RtFunction&& other) : function_(other.function_) {
    other.function_.module = nullptr;
  }
  void operator=(const RtFunction&) = delete;

  std::string name() {
    auto sv = iree_rt_function_name(&function_);
    return std::string(sv.data, sv.size);
  }

  iree_rt_function_signature_t signature() {
    iree_rt_function_signature_t sig;
    CheckApiStatus(iree_rt_function_signature(&function_, &sig),
                   "Error getting function signature");
    return sig;
  }

  iree_rt_function_t& raw_function() { return function_; }

 private:
  iree_rt_function_t function_;
};

class RtModule : public ApiRefCounted<RtModule, iree_rt_module_t> {
 public:
  std::string name() {
    auto sv = iree_rt_module_name(raw_ptr());
    return std::string(sv.data, sv.size);
  }

  absl::optional<RtFunction> lookup_function_by_ordinal(int32_t ordinal) {
    iree_rt_function_t f;
    // TODO(laurenzo): Support an optional linkage argument.
    auto module = raw_ptr();
    auto status = iree_rt_module_lookup_function_by_ordinal(
        module, IREE_RT_FUNCTION_LINKAGE_EXPORT, ordinal, &f);
    if (status == IREE_STATUS_NOT_FOUND) {
      return absl::optional<RtFunction>();
    }
    CheckApiStatus(status, "Error looking up function");
    return RtFunction(f);
  }

  absl::optional<RtFunction> lookup_function_by_name(const std::string& name) {
    iree_rt_function_t f;
    // TODO(laurenzo): Support an optional linkage argument.
    auto module = raw_ptr();
    iree_string_view_t name_sv{name.data(), name.size()};
    auto status = iree_rt_module_lookup_function_by_name(
        module, IREE_RT_FUNCTION_LINKAGE_EXPORT, name_sv, &f);
    if (status == IREE_STATUS_NOT_FOUND) {
      return absl::optional<RtFunction>();
    }
    CheckApiStatus(status, "Error looking up function");
    return RtFunction(f);
  }
};

class RtInstance : public ApiRefCounted<RtInstance, iree_rt_instance_t> {
 public:
  // TODO(laurenzo): Support optional allocator argument.
  static RtInstance Create() {
    iree_rt_instance_t* inst;
    CheckApiStatus(iree_rt_instance_create(IREE_ALLOCATOR_DEFAULT, &inst),
                   "Error creating instance");
    return RtInstance::CreateRetained(inst);
  }
};

class RtPolicy : public ApiRefCounted<RtPolicy, iree_rt_policy_t> {
 public:
  // TODO(laurenzo): Support optional allocator argument.
  static RtPolicy Create() {
    iree_rt_policy_t* policy;
    CheckApiStatus(iree_rt_policy_create(IREE_ALLOCATOR_DEFAULT, &policy),
                   "Error creating policy");
    return RtPolicy::CreateRetained(policy);
  }
};

class RtInvocation : public ApiRefCounted<RtInvocation, iree_rt_invocation_t> {
 public:
  bool QueryStatus() {
    auto status = iree_rt_invocation_query_status(raw_ptr());
    if (status == IREE_STATUS_OK) {
      return true;
    } else if (status == IREE_STATUS_UNAVAILABLE) {
      return false;
    } else {
      CheckApiStatus(status, "Error in function invocation");
      return false;
    }
  }
};

class RtContext : public ApiRefCounted<RtContext, iree_rt_context_t> {
 public:
  static RtContext Create(RtInstance* instance, RtPolicy* policy) {
    iree_rt_context_t* context;
    // TODO(laurenzo): Support optional allocator argument.
    CheckApiStatus(
        iree_rt_context_create(instance->raw_ptr(), policy->raw_ptr(),
                               IREE_ALLOCATOR_DEFAULT, &context),
        "Error creating instance");
    return RtContext::CreateRetained(context);
  }

  int context_id() { return iree_rt_context_id(raw_ptr()); }

  void RegisterModules(std::vector<RtModule*> modules) {
    std::vector<const iree_rt_module_t*> module_raw_ptrs;
    module_raw_ptrs.resize(modules.size());
    for (size_t i = 0, e = modules.size(); i < e; ++i) {
      auto module_raw_ptr = modules[i]->raw_ptr();
      module_raw_ptrs[i] = module_raw_ptr;
    }
    CheckApiStatus(
        iree_rt_context_register_modules(raw_ptr(), module_raw_ptrs.data(),
                                         module_raw_ptrs.size()),
        "Error registering modules");
  }

  void RegisterModule(RtModule* module) {
    const iree_rt_module_t* module_raw_ptr = module->raw_ptr();
    CheckApiStatus(
        iree_rt_context_register_modules(raw_ptr(), &module_raw_ptr, 1),
        "Error registering module");
  }

  absl::optional<RtModule> LookupModuleByName(const std::string& name) {
    iree_rt_module_t* module = iree_rt_context_lookup_module_by_name(
        raw_ptr(), {name.data(), name.size()});
    if (!module) {
      return absl::optional<RtModule>();
    }
    return RtModule::RetainAndCreate(module);
  }

  absl::optional<RtFunction> ResolveFunction(const std::string& full_name) {
    iree_rt_function_t f;
    auto status = iree_rt_context_resolve_function(
        raw_ptr(), {full_name.data(), full_name.size()}, &f);
    if (status == IREE_STATUS_NOT_FOUND) {
      return absl::optional<RtFunction>();
    }
    CheckApiStatus(status, "Error resolving function");
    return RtFunction(f);
  }

  RtInvocation Invoke(RtFunction& f, RtPolicy& policy,
                      std::vector<HalBufferView*> arguments,
                      std::vector<HalBufferView*> results) {
    absl::InlinedVector<const iree_hal_buffer_view_t*, 8> raw_arguments;
    raw_arguments.resize(arguments.size());
    for (size_t i = 0, e = arguments.size(); i < e; ++i) {
      auto inst = arguments[i];
      CheckApiNotNull(inst, "Argument buffer view cannot be None");
      raw_arguments[i] = inst->raw_ptr();
    }
    absl::InlinedVector<const iree_hal_buffer_view_t*, 8> raw_results;
    raw_results.resize(results.size());
    for (size_t i = 0, e = results.size(); i < e; ++i) {
      auto inst = results[i];
      CheckApiNotNull(inst, "Result buffer view cannot be None");
      raw_results[i] = inst->raw_ptr();
    }

    iree_rt_invocation_t* invocation;
    CheckApiStatus(iree_rt_invocation_create(
                       raw_ptr(), &f.raw_function(), policy.raw_ptr(),
                       nullptr /* dependencies */, raw_arguments.data(),
                       raw_arguments.size(), raw_results.data(),
                       raw_results.size(), IREE_ALLOCATOR_DEFAULT, &invocation),
                   "Error invoking function");

    return RtInvocation::CreateRetained(invocation);
  }
};

void SetupRtBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_RT_H_
