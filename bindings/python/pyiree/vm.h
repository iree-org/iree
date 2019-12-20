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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_VM_H_
#define IREE_BINDINGS_PYTHON_PYIREE_VM_H_

#include "absl/types/optional.h"
#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/rt.h"
#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm2/api.h"
#include "iree/vm2/bytecode_module.h"

namespace iree {
namespace python {

//------------------------------------------------------------------------------
// Retain/release bindings
//------------------------------------------------------------------------------

template <>
struct ApiPtrAdapter<iree_vm_instance_t> {
  static void Retain(iree_vm_instance_t* b) { iree_vm_instance_retain(b); }
  static void Release(iree_vm_instance_t* b) { iree_vm_instance_release(b); }
};

template <>
struct ApiPtrAdapter<iree_vm_context_t> {
  static void Retain(iree_vm_context_t* b) { iree_vm_context_retain(b); }
  static void Release(iree_vm_context_t* b) { iree_vm_context_release(b); }
};

template <>
struct ApiPtrAdapter<iree_vm_module_t> {
  static void Retain(iree_vm_module_t* b) { iree_vm_module_retain(b); }
  static void Release(iree_vm_module_t* b) { iree_vm_module_release(b); }
};

template <>
struct ApiPtrAdapter<iree_vm_invocation_t> {
  static void Retain(iree_vm_invocation_t* b) { iree_vm_invocation_retain(b); }
  static void Release(iree_vm_invocation_t* b) {
    iree_vm_invocation_release(b);
  }
};

//------------------------------------------------------------------------------
// ApiRefCounted types
//------------------------------------------------------------------------------

class VmInstance : public ApiRefCounted<VmInstance, iree_vm_instance_t> {
 public:
  static VmInstance Create();
};

class VmModule : public ApiRefCounted<VmModule, iree_vm_module_t> {
 public:
  static VmModule FromFlatbufferBlob(
      std::shared_ptr<OpaqueBlob> flatbuffer_blob);

  absl::optional<iree_vm_function_t> LookupFunction(
      const std::string& name, iree_vm_function_linkage_t linkage);
};

class VmContext : public ApiRefCounted<VmContext, iree_vm_context_t> {
 public:
  // Creates a context, optionally with modules, which will make the context
  // static, disallowing further module registration (and may be more
  // efficient).
  static VmContext Create(VmInstance* instance,
                          absl::optional<std::vector<VmModule*>> modules);

  // Registers additional modules. Only valid for non static contexts (i.e.
  // those created without modules.
  void RegisterModules(std::vector<VmModule*> modules);

  // Unique id for this context.
  int context_id() const { return iree_vm_context_id(raw_ptr()); }
};

class VmInvocation : public ApiRefCounted<VmInvocation, iree_vm_invocation_t> {
};

void SetupVmBindings(pybind11::module m);

//------------------------------------------------------------------------------
// VmVariantList
//------------------------------------------------------------------------------

class VmVariantList {
 public:
  VmVariantList() : list_(nullptr) {}
  ~VmVariantList() {
    if (list_) {
      CheckApiStatus(iree_vm_variant_list_free(list_), "Error freeing list");
    }
  }

  VmVariantList(VmVariantList&& other) {
    list_ = other.list_;
    other.list_ = nullptr;
  }

  VmVariantList& operator=(const VmVariantList&) = delete;
  VmVariantList(const VmVariantList&) = delete;

  static VmVariantList Create(iree_host_size_t capacity) {
    iree_vm_variant_list_t* list;
    CheckApiStatus(
        iree_vm_variant_list_alloc(capacity, IREE_ALLOCATOR_SYSTEM, &list),
        "Error allocating variant list");
    return VmVariantList(list);
  }

  iree_host_size_t size() const { return iree_vm_variant_list_size(list_); }

 private:
  VmVariantList(iree_vm_variant_list_t* list) : list_(list) {}
  iree_vm_variant_list_t* list_;
};

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_VM_H_
