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

#include "bindings/python/pyiree/vm.h"

#include "absl/types/optional.h"
#include "bindings/python/pyiree/function_abi.h"
#include "bindings/python/pyiree/status_utils.h"
#include "iree/base/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm2/invocation.h"
#include "iree/vm2/module.h"

namespace iree {
namespace python {

namespace {

RtModule CreateModuleFromBlob(std::shared_ptr<OpaqueBlob> blob) {
  iree_rt_module_t* module;
  auto free_fn = OpaqueBlob::CreateFreeFn(blob);
  auto status = iree_vm_bytecode_module_create_from_buffer(
      {static_cast<const uint8_t*>(blob->data()), blob->size()}, free_fn.first,
      free_fn.second, IREE_ALLOCATOR_SYSTEM, &module);
  CheckApiStatus(status, "Error creating vm module from blob");
  return RtModule::CreateRetained(module);
}

VmModule CreateHalModule(HalDevice* device) {
  iree_vm_module_t* module;
  CheckApiStatus(
      iree_hal_module_create(device->raw_ptr(), IREE_ALLOCATOR_SYSTEM, &module),
      "Error creating hal module");
  return VmModule::CreateRetained(module);
}

}  // namespace

//------------------------------------------------------------------------------
// VmInstance
//------------------------------------------------------------------------------

VmInstance VmInstance::Create() {
  iree_vm_instance_t* instance;
  auto status = iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance);
  CheckApiStatus(status, "Error creating instance");
  return VmInstance::CreateRetained(instance);
}

//------------------------------------------------------------------------------
// VmContext
//------------------------------------------------------------------------------

VmContext VmContext::Create(VmInstance* instance,
                            absl::optional<std::vector<VmModule*>> modules) {
  iree_vm_context_t* context;
  if (!modules) {
    // Simple create with open allowed modules.
    auto status = iree_vm_context_create(instance->raw_ptr(),
                                         IREE_ALLOCATOR_SYSTEM, &context);
    CheckApiStatus(status, "Error creating vm context");
  } else {
    // Closed set of modules.
    absl::InlinedVector<iree_vm_module_t*, 8> module_handles;
    module_handles.resize(modules->size());
    for (size_t i = 0, e = module_handles.size(); i < e; ++i) {
      module_handles[i] = (*modules)[i]->raw_ptr();
    }
    auto status = iree_vm_context_create_with_modules(
        instance->raw_ptr(), module_handles.data(), module_handles.size(),
        IREE_ALLOCATOR_SYSTEM, &context);
    CheckApiStatus(status, "Error creating vm context with modules");
  }

  CHECK(context);
  return VmContext::CreateRetained(context);
}

void VmContext::RegisterModules(std::vector<VmModule*> modules) {
  absl::InlinedVector<iree_vm_module_t*, 8> module_handles;
  module_handles.resize(modules.size());
  for (size_t i = 0, e = module_handles.size(); i < e; ++i) {
    module_handles[i] = modules[i]->raw_ptr();
  }
  auto status = iree_vm_context_register_modules(raw_ptr(), &module_handles[0],
                                                 module_handles.size());
  CheckApiStatus(status, "Error registering modules");
}

std::unique_ptr<FunctionAbi> VmContext::CreateFunctionAbi(
    HalDevice& device, std::shared_ptr<HostTypeFactory> host_type_factory,
    iree_vm_function_t f) {
  // Resolve attrs.
  absl::InlinedVector<std::pair<iree_string_view_t, iree_string_view_t>, 4>
      attrs;
  for (int i = 0;; ++i) {
    attrs.push_back({});
    auto status = iree_vm_get_function_reflection_attr(
        f, i, &attrs.back().first, &attrs.back().second);
    if (status == IREE_STATUS_NOT_FOUND) {
      attrs.pop_back();
      break;
    }
    CheckApiStatus(status, "Error getting reflection attr");
  }
  auto attr_lookup =
      [&attrs](absl::string_view key) -> absl::optional<absl::string_view> {
    for (const auto& attr : attrs) {
      absl::string_view found_key(attr.first.data, attr.first.size);
      absl::string_view found_value(attr.second.data, attr.second.size);
      if (found_key == key) return found_value;
    }
    return absl::nullopt;
  };

  return FunctionAbi::Create(device, std::move(host_type_factory), attr_lookup);
}

void VmContext::Invoke(iree_vm_function_t f, VmVariantList& inputs,
                       VmVariantList& outputs) {
  CheckApiStatus(iree_vm_invoke(raw_ptr(), f, nullptr, inputs.raw_ptr(),
                                outputs.raw_ptr(), IREE_ALLOCATOR_SYSTEM),
                 "Error invoking function");
}

//------------------------------------------------------------------------------
// VmModule
//------------------------------------------------------------------------------

VmModule VmModule::FromFlatbufferBlob(
    std::shared_ptr<OpaqueBlob> flatbuffer_blob) {
  iree_vm_module_t* module;
  auto deallocator = OpaqueBlob::CreateDeallocator(flatbuffer_blob);
  auto status = iree_vm_bytecode_module_create(
      {static_cast<const uint8_t*>(flatbuffer_blob->data()),
       flatbuffer_blob->size()},
      deallocator, IREE_ALLOCATOR_SYSTEM, &module);
  if (status != IREE_STATUS_OK) {
    deallocator.free(deallocator.self, nullptr);
  }

  CheckApiStatus(status, "Error creating vm module from flatbuffer");
  return VmModule::CreateRetained(module);
}

absl::optional<iree_vm_function_t> VmModule::LookupFunction(
    const std::string& name, iree_vm_function_linkage_t linkage) {
  iree_vm_function_t f;
  auto status = iree_vm_module_lookup_function_by_name(
      raw_ptr(), linkage, {name.data(), name.size()}, &f);
  if (status == IREE_STATUS_NOT_FOUND) {
    return absl::nullopt;
  }
  CheckApiStatus(status, "Error looking up function");
  return f;
}

//------------------------------------------------------------------------------
// VmVariantList
//------------------------------------------------------------------------------

std::string VmVariantList::DebugString() const {
  // The variant list API requires mutability, so we const cast to it internally
  // so we can maintain a const DebugString() for callers.
  auto mutable_this = const_cast<VmVariantList*>(this);
  std::string s;
  absl::StrAppend(&s, "<VmVariantList(", size(), "): [");

  for (iree_host_size_t i = 0, e = size(); i < e; ++i) {
    iree_vm_variant_t* variant =
        iree_vm_variant_list_get(mutable_this->raw_ptr(), i);
    if (i > 0) absl::StrAppend(&s, ", ");

    if (IREE_VM_VARIANT_IS_VALUE(variant)) {
      absl::StrAppend(&s, variant->i32);
    } else if (IREE_VM_VARIANT_IS_REF(variant)) {
      // Pretty print a subset of ABI impacting known types.
      if (iree_hal_buffer_isa(&variant->ref)) {
        auto* hal_buffer = iree_hal_buffer_deref(&variant->ref);
        assert(hal_buffer);
        absl::StrAppend(&s, "HalBuffer(",
                        iree_hal_buffer_byte_length(hal_buffer), ")");
      } else {
        absl::StrAppend(&s, "Unknown(", variant->ref_type, ")");
      }
    } else {
      absl::StrAppend(&s, "None");
    }
  }
  absl::StrAppend(&s, "]>");
  return s;
}

void SetupVmBindings(pybind11::module m) {
  CHECK_EQ(IREE_STATUS_OK, iree_vm_register_builtin_types());
  CHECK_EQ(IREE_STATUS_OK, iree_hal_module_register_types());

  // Deprecated: VM1 module.
  m.def("create_module_from_blob", CreateModuleFromBlob);

  // Built-in module creation.
  m.def("create_hal_module", &CreateHalModule);

  py::enum_<iree_vm_function_linkage_t>(m, "Linkage")
      .value("INTERNAL", IREE_VM_FUNCTION_LINKAGE_INTERNAL)
      .value("IMPORT", IREE_VM_FUNCTION_LINKAGE_IMPORT)
      .value("EXPORT", IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .export_values();

  // Mutation and inspection of the variant list is mostly opaque to python.
  py::class_<VmVariantList>(m, "VmVariantList")
      .def(py::init(&VmVariantList::Create))
      .def_property_readonly("size", &VmVariantList::size)
      .def("__repr__", &VmVariantList::DebugString);

  py::class_<iree_vm_function_t>(m, "VmFunction")
      .def_readonly("linkage", &iree_vm_function_t::linkage)
      .def_readonly("ordinal", &iree_vm_function_t::ordinal);

  py::class_<VmInstance>(m, "VmInstance").def(py::init(&VmInstance::Create));

  py::class_<VmContext>(m, "VmContext")
      .def(py::init(&VmContext::Create), py::arg("instance"),
           py::arg("modules") = absl::nullopt)
      .def("register_modules", &VmContext::RegisterModules)
      .def_property_readonly("context_id", &VmContext::context_id)
      .def("create_function_abi", &VmContext::CreateFunctionAbi,
           py::arg("device"), py::arg("host_type_factory"), py::arg("f"))
      .def("invoke", &VmContext::Invoke);

  py::class_<VmModule>(m, "VmModule")
      .def_static("from_flatbuffer", &VmModule::FromFlatbufferBlob)
      .def("lookup_function", &VmModule::LookupFunction, py::arg("name"),
           py::arg("linkage") = IREE_VM_FUNCTION_LINKAGE_EXPORT);
}

}  // namespace python
}  // namespace iree
