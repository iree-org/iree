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

#include "iree/bindings/python/pyiree/binding.h"
#include "iree/bindings/python/pyiree/status_utils.h"
#include "iree/rt/api.h"

namespace iree {
namespace python {

template <>
struct ApiPtrAdapter<iree_rt_module_t> {
  static void Retain(iree_rt_module_t* m) { iree_rt_module_retain(m); }
  static void Release(iree_rt_module_t* m) { iree_rt_module_release(m); }
};

class RtFunction {
 public:
  // Note that this will retain the module.
  RtFunction(iree_rt_module* module, iree_rt_function_t function)
      : module_(module), function_(function) {
    iree_rt_module_retain(module_);
  }
  ~RtFunction() { iree_rt_module_release(module_); }

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

 private:
  iree_rt_module* module_;
  iree_rt_function_t function_;
};

class RtModule : public ApiRefCounted<RtModule, iree_rt_module_t> {
 public:
  std::string name() {
    auto sv = iree_rt_module_name(instance());
    return std::string(sv.data, sv.size);
  }

  std::unique_ptr<RtFunction> lookup_function_by_ordinal(int32_t ordinal) {
    iree_rt_function_t f;
    // TODO(laurenzo): Support an optional linkage argument.
    auto module = instance();
    auto status = iree_rt_module_lookup_function_by_ordinal(
        module, IREE_RT_FUNCTION_LINKAGE_EXPORT, ordinal, &f);
    if (status == IREE_STATUS_NOT_FOUND) {
      return nullptr;
    }
    CheckApiStatus(status, "Error looking up function");
    return std::make_unique<RtFunction>(module, f);
  }

  std::unique_ptr<RtFunction> lookup_function_by_name(const std::string& name) {
    iree_rt_function_t f;
    // TODO(laurenzo): Support an optional linkage argument.
    auto module = instance();
    iree_string_view_t name_sv{name.data(), name.size()};
    auto status = iree_rt_module_lookup_function_by_name(
        module, IREE_RT_FUNCTION_LINKAGE_EXPORT, name_sv, &f);
    if (status == IREE_STATUS_NOT_FOUND) {
      return nullptr;
    }
    CheckApiStatus(status, "Error looking up function");
    return std::make_unique<RtFunction>(module, f);
  }
};

void SetupRtBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_RT_H_
