// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_VM_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_VM_H_

#include <optional>

#include "./binding.h"
#include "./status_utils.h"
#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

namespace iree {
namespace python {

class FunctionAbi;

//------------------------------------------------------------------------------
// Retain/release bindings
//------------------------------------------------------------------------------

template <>
struct ApiPtrAdapter<iree_vm_buffer_t> {
  static void Retain(iree_vm_buffer_t* b) { iree_vm_buffer_retain(b); }
  static void Release(iree_vm_buffer_t* b) { iree_vm_buffer_release(b); }
};

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
struct ApiPtrAdapter<iree_vm_list_t> {
  static void Retain(iree_vm_list_t* b) { iree_vm_list_retain(b); }
  static void Release(iree_vm_list_t* b) { iree_vm_list_release(b); }
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

template <>
struct ApiPtrAdapter<iree_vm_ref_t> {
  static void Retain(iree_vm_ref_t* b) {
    iree_vm_ref_t out_ref;
    std::memset(&out_ref, 0, sizeof(out_ref));
    iree_vm_ref_retain(b, &out_ref);
  }
  static void Release(iree_vm_ref_t* b) { iree_vm_ref_release(b); }
};

//------------------------------------------------------------------------------
// VmBuffer
//------------------------------------------------------------------------------

class VmBuffer : public ApiRefCounted<VmBuffer, iree_vm_buffer_t> {};

//------------------------------------------------------------------------------
// VmVariantList
// TODO: Rename to VmList
//------------------------------------------------------------------------------

class VmVariantList : public ApiRefCounted<VmVariantList, iree_vm_list_t> {
 public:
  static VmVariantList Create(iree_host_size_t capacity) {
    iree_vm_list_t* list;
    CheckApiStatus(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), capacity,
                            iree_allocator_system(), &list),
        "Error allocating variant list");
    return VmVariantList::StealFromRawPtr(list);
  }

  iree_host_size_t size() const { return iree_vm_list_size(raw_ptr()); }

  void AppendNullRef() {
    iree_vm_ref_t null_ref = {0};
    CheckApiStatus(iree_vm_list_push_ref_move(raw_ptr(), &null_ref),
                   "Error appending to list");
  }

  std::string DebugString() const;
  void PushFloat(double fvalue);
  void PushInt(int64_t ivalue);
  void PushList(VmVariantList& other);
  void PushRef(py::handle ref_or_object);
  py::object GetAsList(int index);
  py::object GetAsRef(int index);
  py::object GetAsObject(int index, py::object clazz);
  py::object GetVariant(int index);
  py::object GetAsSerializedTraceValue(int index);
};

//------------------------------------------------------------------------------
// VmInstance
//------------------------------------------------------------------------------

class VmInstance : public ApiRefCounted<VmInstance, iree_vm_instance_t> {
 public:
  static VmInstance Create();
};

//------------------------------------------------------------------------------
// VmModule
//------------------------------------------------------------------------------

class VmModule : public ApiRefCounted<VmModule, iree_vm_module_t> {
 public:
  static VmModule ResolveModuleDependency(VmInstance* instance,
                                          const std::string& name,
                                          uint32_t minimum_version);

  static VmModule FromFlatbufferBlob(VmInstance* instance,
                                     py::object flatbuffer_blob_object);

  std::optional<iree_vm_function_t> LookupFunction(
      const std::string& name, iree_vm_function_linkage_t linkage);

  std::string name() const {
    auto name_sv = iree_vm_module_name(raw_ptr());
    return std::string(name_sv.data, name_sv.size);
  }

  py::object get_stashed_flatbuffer_blob() { return stashed_flatbuffer_blob; }

 private:
  // If the module was created from a FlatBuffer blob, we stash it here.
  py::object stashed_flatbuffer_blob = py::none();
};

//------------------------------------------------------------------------------
// VmContext
//------------------------------------------------------------------------------

class VmContext : public ApiRefCounted<VmContext, iree_vm_context_t> {
 public:
  // Creates a context, optionally with modules, which will make the context
  // static, disallowing further module registration (and may be more
  // efficient).
  static VmContext Create(VmInstance* instance,
                          std::optional<std::vector<VmModule*>> modules);

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

//------------------------------------------------------------------------------
// VmRef (represents a pointer to an arbitrary reference object).
//------------------------------------------------------------------------------

class VmRef {
 public:
  //----------------------------------------------------------------------------
  // Binds the reference protocol to a VmRefObject bound class.
  // This defines three attributes:
  //   __iree_vm_type__()
  //        Gets the type from the object.
  //   [readonly property] __iree_vm_ref__ :
  //        Gets a VmRef from the object.
  //   __iree_vm_cast__(ref) :
  //        Dereferences the VmRef to the concrete type. Returns None on cast
  //        failure.
  //
  // In addition, a user attribute of "ref" will be added that is an alias of
  // __iree_vm_ref__.
  //
  // An __eq__ method is added which returns true if the python objects refer
  // to the same vm object.
  //
  // The BindRefProtocol() helper is used on a py::class_ defined for a
  // reference object. It takes some of the C helper functions that are defined
  // for each type and is generic.
  //----------------------------------------------------------------------------
  static const char* const kTypeAttr;
  static const char* const kRefAttr;
  static const char* const kCastAttr;

  template <typename PyClass, typename TypeFunctor, typename RetainRefFunctor,
            typename DerefFunctor, typename IsaFunctor>
  static void BindRefProtocol(PyClass& cls, TypeFunctor type,
                              RetainRefFunctor retain_ref, DerefFunctor deref,
                              IsaFunctor isa) {
    using WrapperType = typename PyClass::type;
    using RawPtrType = typename WrapperType::RawPtrType;
    auto ref_lambda = [=](WrapperType& self) {
      return VmRef::Steal(retain_ref(self.raw_ptr()));
    };
    cls.def_static(VmRef::kTypeAttr, [=]() { return type(); });
    cls.def_property_readonly(VmRef::kRefAttr, ref_lambda);
    cls.def_property_readonly("ref", ref_lambda);
    cls.def_static(VmRef::kCastAttr, [=](VmRef& ref) -> py::object {
      if (!isa(ref.ref())) {
        return py::none();
      }
      return py::cast(WrapperType::BorrowFromRawPtr(deref(ref.ref())),
                      py::return_value_policy::move);
    });
    cls.def("__eq__", [](WrapperType& self, WrapperType& other) {
      return self.raw_ptr() == other.raw_ptr();
    });
    cls.def("__eq__",
            [](WrapperType& self, py::object& other) { return false; });
  }

  // Initializes a null ref.
  VmRef() { std::memset(&ref_, 0, sizeof(ref_)); }
  VmRef(VmRef&& other) : ref_(other.ref_) {
    std::memset(&other.ref_, 0, sizeof(other.ref_));
  }
  VmRef(const VmRef&) = delete;
  VmRef& operator=(const VmRef&) = delete;
  ~VmRef() {
    if (ref_.ptr) {
      iree_vm_ref_release(&ref_);
    }
  }

  // Creates a VmRef from an owned ref, taking the reference count.
  static VmRef Steal(iree_vm_ref_t ref) { return VmRef(ref); }

  iree_vm_ref_t& ref() { return ref_; }

  py::object Deref(py::object ref_object_class, bool optional);
  bool IsInstance(py::object ref_object_class);

  std::string ToString();

 private:
  // Initializes with an owned ref.
  VmRef(iree_vm_ref_t ref) : ref_(ref) {}
  iree_vm_ref_t ref_;
};

void SetupVmBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_VM_H_
