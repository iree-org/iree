// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_VM_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_VM_H_

#include "absl/types/optional.h"
#include "bindings/python/iree/runtime/binding.h"
#include "bindings/python/iree/runtime/hal.h"
#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

namespace iree {
namespace python {

class FunctionAbi;

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
// VmVariantList
//------------------------------------------------------------------------------

class VmVariantList {
 public:
  VmVariantList() : list_(nullptr) {}
  ~VmVariantList() {
    if (list_) {
      iree_vm_list_release(list_);
    }
  }

  VmVariantList(VmVariantList&& other) {
    list_ = other.list_;
    other.list_ = nullptr;
  }

  VmVariantList& operator=(const VmVariantList&) = delete;
  VmVariantList(const VmVariantList&) = delete;

  static VmVariantList Create(iree_host_size_t capacity) {
    iree_vm_list_t* list;
    CheckApiStatus(iree_vm_list_create(/*element_type=*/nullptr, capacity,
                                       iree_allocator_system(), &list),
                   "Error allocating variant list");
    return VmVariantList(list);
  }

  iree_host_size_t size() const { return iree_vm_list_size(list_); }

  iree_vm_list_t* raw_ptr() { return list_; }
  const iree_vm_list_t* raw_ptr() const { return list_; }

  void AppendNullRef() {
    iree_vm_ref_t null_ref = {0};
    CheckApiStatus(iree_vm_list_push_ref_move(raw_ptr(), &null_ref),
                   "Error appending to list");
  }

  std::string DebugString() const;
  void PushFloat(double fvalue);
  void PushInt(int64_t ivalue);
  void PushList(VmVariantList& other);
  void PushBufferView(HalDevice& device, py::object py_buffer_object,
                      iree_hal_element_type_t element_type);
  py::object GetAsList(int index);
  py::object GetAsNdarray(int index);
  py::object GetVariant(int index);

 private:
  VmVariantList(iree_vm_list_t* list) : list_(list) {}
  iree_vm_list_t* list_;
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
  static VmModule FromFlatbufferBlob(py::buffer flatbuffer_blob);

  absl::optional<iree_vm_function_t> LookupFunction(
      const std::string& name, iree_vm_function_linkage_t linkage);

  std::string name() const {
    auto name_sv = iree_vm_module_name(raw_ptr());
    return std::string(name_sv.data, name_sv.size);
  }
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

  // Synchronously invokes the given function.
  void Invoke(iree_vm_function_t f, VmVariantList& inputs,
              VmVariantList& outputs);
};

class VmInvocation : public ApiRefCounted<VmInvocation, iree_vm_invocation_t> {
};

void SetupVmBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_VM_H_
