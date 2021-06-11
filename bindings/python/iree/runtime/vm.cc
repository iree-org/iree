// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bindings/python/iree/runtime/vm.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "bindings/python/iree/runtime/function_abi.h"
#include "bindings/python/iree/runtime/status_utils.h"
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/strings_module.h"
#include "iree/modules/tensorlist/native_module.h"
#include "iree/vm/api.h"
#include "pybind11/numpy.h"

namespace iree {
namespace python {

namespace {

VmModule CreateHalModule(HalDevice* device) {
  iree_vm_module_t* module;
  CheckApiStatus(iree_hal_module_create(device->raw_ptr(),
                                        iree_allocator_system(), &module),
                 "Error creating hal module");
  return VmModule::CreateRetained(module);
}

VmModule CreateStringsModule() {
  iree_vm_module_t* module;
  CheckApiStatus(iree_strings_module_create(iree_allocator_system(), &module),
                 "Error creating trings module");
  return VmModule::CreateRetained(module);
}

VmModule CreateTensorListModule() {
  iree_vm_module_t* module;
  CheckApiStatus(
      iree_tensorlist_module_create(iree_allocator_system(), &module),
      "Error creating tensorlist module");
  return VmModule::CreateRetained(module);
}

// RAII wrapper for a Py_buffer which calls PyBuffer_Release when it goes
// out of scope.
class PyBufferReleaser {
 public:
  PyBufferReleaser(Py_buffer& b) : b_(b) {}
  ~PyBufferReleaser() { PyBuffer_Release(&b_); }

 private:
  Py_buffer& b_;
};

py::dict GetFunctionReflectionDict(iree_vm_function_t& f) {
  py::dict attrs;
  for (int i = 0;; ++i) {
    iree_string_view_t key;
    iree_string_view_t value;
    auto status = iree_vm_get_function_reflection_attr(f, i, &key, &value);
    if (iree_status_is_not_found(status)) {
      iree_status_ignore(status);
      break;
    }
    CheckApiStatus(status, "Error getting reflection attr");
    py::str key_str(key.data, key.size);
    py::str value_str(value.data, value.size);
    attrs[std::move(key_str)] = std::move(value_str);
  }
  return attrs;
}

}  // namespace

//------------------------------------------------------------------------------
// VmInstance
//------------------------------------------------------------------------------

VmInstance VmInstance::Create() {
  iree_vm_instance_t* instance;
  auto status = iree_vm_instance_create(iree_allocator_system(), &instance);
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
                                         iree_allocator_system(), &context);
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
        iree_allocator_system(), &context);
    CheckApiStatus(status, "Error creating vm context with modules");
  }

  IREE_CHECK(context);
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
    if (iree_status_is_not_found(status)) {
      iree_status_ignore(status);
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
                                outputs.raw_ptr(), iree_allocator_system()),
                 "Error invoking function");
}

//------------------------------------------------------------------------------
// VmModule
//------------------------------------------------------------------------------

VmModule VmModule::FromFlatbufferBlob(py::buffer flatbuffer_blob) {
  auto buffer_info = flatbuffer_blob.request();
  iree_vm_module_t* module;

  // Bridge to the C-based deallocator API.
  auto* raw_ptr = flatbuffer_blob.ptr();
  auto free_fn = +([](void* self, void*) {
    PyObject* self_ptr = static_cast<PyObject*>(self);
    Py_XDECREF(self_ptr);
  });
  flatbuffer_blob.inc_ref();
  iree_allocator_t deallocator{raw_ptr /* self */, nullptr /* alloc */,
                               free_fn /* dealloc */};

  auto status = iree_vm_bytecode_module_create(
      {static_cast<const uint8_t*>(buffer_info.ptr),
       static_cast<iree_host_size_t>(buffer_info.size)},
      deallocator, iree_allocator_system(), &module);
  if (!iree_status_is_ok(status)) {
    deallocator.free(raw_ptr, nullptr);
  }

  CheckApiStatus(status, "Error creating vm module from flatbuffer");
  return VmModule::CreateRetained(module);
}

absl::optional<iree_vm_function_t> VmModule::LookupFunction(
    const std::string& name, iree_vm_function_linkage_t linkage) {
  iree_vm_function_t f;
  auto status = iree_vm_module_lookup_function_by_name(
      raw_ptr(), linkage, {name.data(), name.size()}, &f);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return absl::nullopt;
  }
  CheckApiStatus(status, "Error looking up function");
  return f;
}

//------------------------------------------------------------------------------
// VmVariantList
//------------------------------------------------------------------------------

void VmVariantList::PushFloat(double fvalue) {
  // Note that Python floats are f64.
  iree_vm_value_t value = iree_vm_value_make_f64(fvalue);
  CheckApiStatus(iree_vm_list_push_value(raw_ptr(), &value),
                 "Could not push float");
}

void VmVariantList::PushInt(int64_t ivalue) {
  // Note that Python ints are unbounded, so just use the largest type we
  // have.
  iree_vm_value_t value = iree_vm_value_make_i64(ivalue);
  CheckApiStatus(iree_vm_list_push_value(raw_ptr(), &value),
                 "Could not push int");
}

void VmVariantList::PushList(VmVariantList& other) {
  iree_vm_ref_t retained = iree_vm_list_retain_ref(other.raw_ptr());
  iree_vm_list_push_ref_move(raw_ptr(), &retained);
}

void VmVariantList::PushBufferView(HalDevice& device,
                                   py::object py_buffer_object,
                                   iree_hal_element_type_t element_type) {
  // Request a view of the buffer (use the raw python C API to avoid some
  // allocation and copying at the pybind level).
  Py_buffer py_view;
  // Note that only C-Contiguous ND-arrays are presently supported, so
  // only request that via PyBUF_ND. Long term, we should consult an
  // "oracle" in the runtime to determine the precise required format and
  // set flags accordingly (and fallback/copy on failure).
  int flags = PyBUF_FORMAT | PyBUF_ND;

  // Acquire the backing buffer and setup RAII release.
  if (PyObject_GetBuffer(py_buffer_object.ptr(), &py_view, flags) != 0) {
    // The GetBuffer call is required to set an appropriate error.
    throw py::error_already_set();
  }
  PyBufferReleaser py_view_releaser(py_view);

  // Whether the py object needs to be retained with the argument.
  // Should be set to true if directly mapping, false if copied.
  bool depends_on_pyobject = false;

  // Allocate a HalBuffer.
  // This is hard-coded to C-contiguous right now.
  // TODO(laurenzo): Expand to other layouts as needed.
  // TODO(laurenzo): Wrap and retain original buffer (depends_on_pyobject=true).
  iree_hal_buffer_t* raw_buffer;
  CheckApiStatus(iree_hal_allocator_allocate_buffer(
                     device.allocator(),
                     static_cast<iree_hal_memory_type_t>(
                         IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                         IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
                     IREE_HAL_BUFFER_USAGE_ALL, py_view.len, &raw_buffer),
                 "Failed to allocate device visible buffer");
  CheckApiStatus(
      iree_hal_buffer_write_data(raw_buffer, 0, py_view.buf, py_view.len),
      "Error writing to input buffer");

  // Only capture the reference to the exporting object (incrementing it)
  // once guaranteed successful.
  if (depends_on_pyobject) {
    // Note for future implementation: there needs to be a place to stash
    // references to be kept alive which back a buffer. This is likely an
    // additional bag of refs returned from this function, which can then
    // be attached to an invocation.
    throw RaisePyError(PyExc_NotImplementedError,
                       "Dependent buffer arguments not implemented");
  }

  // Create the buffer_view. (note that numpy shape is ssize_t)
  std::vector<int> dims(py_view.ndim);
  std::copy(py_view.shape, py_view.shape + py_view.ndim, dims.begin());
  iree_hal_buffer_view_t* buffer_view;
  CheckApiStatus(
      iree_hal_buffer_view_create(raw_buffer, dims.data(), dims.size(),
                                  element_type, &buffer_view),
      "Error allocating buffer_view");
  iree_hal_buffer_release(raw_buffer);
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  CheckApiStatus(iree_vm_list_push_ref_move(raw_ptr(), &buffer_view_ref),
                 "Error moving buffer view");
}

py::object VmVariantList::GetAsList(int index) {
  iree_vm_ref_t ref = {0};
  CheckApiStatus(iree_vm_list_get_ref_assign(raw_ptr(), index, &ref),
                 "Could not access list element");
  iree_vm_list_t* sub_list = NULL;
  CheckApiStatus(iree_vm_list_check_deref(ref, &sub_list),
                 "Could not deref list (wrong type?)");
  iree_vm_list_retain(sub_list);
  return py::cast(VmVariantList(sub_list));
}

py::object VmVariantList::GetVariant(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant(raw_ptr(), index, &v),
                 "Could not access list element");
  if (iree_vm_type_def_is_value(&v.type)) {
    // Convert a value type.
    switch (v.type.value_type) {
      case IREE_VM_VALUE_TYPE_I8:
        return py::cast(v.i8);
      case IREE_VM_VALUE_TYPE_I16:
        return py::cast(v.i16);
      case IREE_VM_VALUE_TYPE_I32:
        return py::cast(v.i32);
      case IREE_VM_VALUE_TYPE_I64:
        return py::cast(v.i64);
      case IREE_VM_VALUE_TYPE_F32:
        return py::cast(v.f32);
      case IREE_VM_VALUE_TYPE_F64:
        return py::cast(v.f64);
      default:
        throw RaiseValueError("Unsupported VM value type conversion");
    }
  } else if (v.type.ref_type == IREE_VM_REF_TYPE_NULL) {
    return py::none();
  } else if (iree_vm_type_def_is_ref(&v.type)) {
    // Convert reference type.
    if (iree_vm_list_isa(v.ref)) {
      return GetAsList(index);
    } else if (iree_hal_buffer_view_isa(v.ref)) {
      return GetAsNdarray(index);
    }
  }

  throw RaiseValueError("Unsupported VM to Python Type Conversion");
}

py::object VmVariantList::GetAsNdarray(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant(raw_ptr(), index, &v),
                 "Could not access list element");
  iree_hal_buffer_view_t* buffer_view = iree_hal_buffer_view_deref(v.ref);
  if (!buffer_view) {
    throw RaiseValueError("Could not deref result buffer view (wrong type?)");
  }
  iree_hal_buffer_t* raw_buffer = iree_hal_buffer_view_buffer(buffer_view);
  if (!raw_buffer) {
    throw RaiseValueError("Could not deref result buffer (wrong type?)");
  }
  HalBuffer buffer = HalBuffer::RetainAndCreate(raw_buffer);

  // Extract dims from the buffer view.
  size_t rank = 0;
  std::vector<int32_t> dims(6);
  iree_status_t status = iree_hal_buffer_view_shape(
      buffer_view, dims.capacity(), dims.data(), &rank);
  if (iree_status_is_out_of_range(status)) {
    dims.resize(rank);
    status = iree_hal_buffer_view_shape(buffer_view, dims.capacity(),
                                        dims.data(), &rank);
  }
  CheckApiStatus(status, "Error extracting shape");
  dims.resize(rank);

  // Convert element type to dtype.
  iree_hal_element_type_t element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  // See: https://docs.python.org/3/c-api/arg.html#numbers
  // TODO: Handle dtypes that do not map to a code (i.e. fp16).
  const char* dtype_code;
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      dtype_code = "b";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      dtype_code = "B";
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      dtype_code = "h";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      dtype_code = "H";
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      dtype_code = "i";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      dtype_code = "I";
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      dtype_code = "l";
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      dtype_code = "L";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      dtype_code = "f";
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      dtype_code = "d";
      break;
    case IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 1):
      // Due to layering issues it is not uncommon to get i1 buffer views
      // and we just silently promote them to i8 since that is what they are.
      // Really i1 should not exist at this boundary.
      dtype_code = "b";
      break;
    default:
      throw RaiseValueError("Unsupported VM Buffer -> numpy dtype mapping");
  }
  auto dtype = py::dtype(dtype_code);

  // Map memory.
  iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(buffer.raw_ptr());
  iree_hal_buffer_mapping_t mapped_memory;
  CheckApiStatus(iree_hal_buffer_map_range(
                     buffer.raw_ptr(), IREE_HAL_MEMORY_ACCESS_READ,
                     0 /* element_offset */, byte_length, &mapped_memory),
                 "Could not map memory");

  // Turn the mapping into a python object that retains until the array is
  // destroyed.
  HalMappedMemory hal_mapped_memory(mapped_memory, buffer_view);
  py::object py_mapped_memory = py::cast(
      std::move(hal_mapped_memory), py::return_value_policy::take_ownership);
  return py::array(std::move(dtype), dims, mapped_memory.contents.data,
                   std::move(py_mapped_memory) /* base */);
}

namespace {

void AppendListContents(std::string& out, iree_vm_list_t* list,
                        std::unordered_set<iree_vm_list_t*>& visited) {
  for (iree_host_size_t i = 0, e = iree_vm_list_size(list); i < e; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    iree_status_t status = iree_vm_list_get_variant(list, i, &variant);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      out.append("Error");
      continue;
    }
    if (i > 0) out.append(", ");

    if (iree_vm_variant_is_value(variant)) {
      absl::StrAppend(&out, variant.i32);
    } else if (iree_vm_variant_is_ref(variant)) {
      // Pretty print a subset of ABI impacting known types.
      if (iree_hal_buffer_isa(variant.ref)) {
        auto* hal_buffer = iree_hal_buffer_deref(variant.ref);
        assert(hal_buffer);
        absl::StrAppend(&out, "HalBuffer(",
                        iree_hal_buffer_byte_length(hal_buffer), ")");
      } else if (iree_hal_buffer_view_isa(variant.ref)) {
        auto hal_bv = iree_hal_buffer_view_deref(variant.ref);
        absl::StrAppend(&out, "HalBufferView(");
        absl::InlinedVector<int32_t, 5> shape(
            iree_hal_buffer_view_shape_rank(hal_bv));
        iree_hal_buffer_view_shape(hal_bv, shape.size(), shape.data(), nullptr);
        absl::StrAppend(&out, absl::StrJoin(shape, "x"), ":0x",
                        absl::Hex(static_cast<uint32_t>(
                            iree_hal_buffer_view_element_type(hal_bv))),
                        ")");
      } else if (iree_vm_list_isa(variant.ref)) {
        out.append("List[");
        iree_vm_list_t* sub_list = iree_vm_list_deref(variant.ref);
        if (visited.insert(sub_list).second) {
          AppendListContents(out, sub_list, visited);
        } else {
          out.append("...circular...");
        }
        out.append("]");
      } else {
        absl::StrAppend(&out, "Unknown(", variant.type.ref_type, ")");
      }
    } else {
      out.append("None");
    }
  }
}

}  // namespace

std::string VmVariantList::DebugString() const {
  // The variant list API requires mutability, so we const cast to it internally
  // so we can maintain a const DebugString() for callers.
  auto mutable_this = const_cast<VmVariantList*>(this);
  std::string s;
  absl::StrAppend(&s, "<VmVariantList(", size(), "): [");
  iree_vm_list_t* list = mutable_this->raw_ptr();
  std::unordered_set<iree_vm_list_t*> visited;
  visited.insert(list);
  AppendListContents(s, list, visited);
  s.append("]>");
  return s;
}

void SetupVmBindings(pybind11::module m) {
  IREE_CHECK_OK(iree_vm_register_builtin_types());
  IREE_CHECK_OK(iree_hal_module_register_types());
  IREE_CHECK_OK(iree_tensorlist_module_register_types());
  IREE_CHECK_OK(iree_strings_module_register_types());

  // Built-in module creation.
  m.def("create_hal_module", &CreateHalModule);
  m.def("create_strings_module", &CreateStringsModule);
  m.def("create_tensorlist_module", &CreateTensorListModule);

  py::enum_<enum iree_vm_function_linkage_e>(m, "Linkage")
      .value("INTERNAL", IREE_VM_FUNCTION_LINKAGE_INTERNAL)
      .value("IMPORT", IREE_VM_FUNCTION_LINKAGE_IMPORT)
      .value("EXPORT", IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .export_values();

  // Mutation and inspection of the variant list is mostly opaque to python.
  py::class_<VmVariantList>(m, "VmVariantList")
      .def(py::init(&VmVariantList::Create))
      .def_property_readonly("size", &VmVariantList::size)
      .def("__len__", &VmVariantList::size)
      .def("get_as_ndarray", &VmVariantList::GetAsNdarray)
      .def("get_as_list", &VmVariantList::GetAsList)
      .def("get_variant", &VmVariantList::GetVariant)
      .def("push_float", &VmVariantList::PushFloat)
      .def("push_int", &VmVariantList::PushInt)
      .def("push_list", &VmVariantList::PushList)
      .def("push_buffer_view", &VmVariantList::PushBufferView)
      .def("__repr__", &VmVariantList::DebugString);

  py::class_<iree_vm_function_t>(m, "VmFunction")
      .def_readonly("linkage", &iree_vm_function_t::linkage)
      .def_readonly("ordinal", &iree_vm_function_t::ordinal)
      .def_property_readonly("reflection",
                             [](iree_vm_function_t& self) {
                               return GetFunctionReflectionDict(self);
                             })
      .def("__repr__", [](iree_vm_function_t& self) {
        iree_string_view_t name = iree_vm_function_name(&self);
        std::string repr("<VmFunction ");
        repr.append(name.data, name.size);

        iree_vm_function_signature_t sig = iree_vm_function_signature(&self);
        repr.append("(");
        repr.append(sig.calling_convention.data, sig.calling_convention.size);
        repr.append("), reflection = ");
        py::dict reflection = GetFunctionReflectionDict(self);
        repr.append(py::cast<std::string>(py::repr(reflection)));
        repr.append(">");
        return repr;
      });

  py::class_<VmInstance>(m, "VmInstance").def(py::init(&VmInstance::Create));

  py::class_<VmContext>(m, "VmContext")
      .def(py::init(&VmContext::Create), py::arg("instance"),
           py::arg("modules") = absl::optional<std::vector<VmModule*>>())
      .def("register_modules", &VmContext::RegisterModules)
      .def_property_readonly("context_id", &VmContext::context_id)
      .def("create_function_abi", &VmContext::CreateFunctionAbi,
           py::arg("device"), py::arg("host_type_factory"), py::arg("f"))
      .def("invoke", &VmContext::Invoke);

  py::class_<VmModule>(m, "VmModule")
      .def_static("from_flatbuffer", &VmModule::FromFlatbufferBlob)
      .def_property_readonly("name", &VmModule::name)
      .def("lookup_function", &VmModule::LookupFunction, py::arg("name"),
           py::arg("linkage") = IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .def("__repr__", [](VmModule& self) {
        std::string repr("<VmModule ");
        iree_string_view_t name = iree_vm_module_name(self.raw_ptr());
        repr.append(name.data, name.size);

        iree_vm_module_signature_t sig =
            iree_vm_module_signature(self.raw_ptr());
        repr.append(" : [");
        for (size_t ordinal = 0; ordinal < sig.export_function_count;
             ++ordinal) {
          iree_vm_function_t f;
          iree_string_view_t linkage_name;
          auto status = iree_vm_module_lookup_function_by_ordinal(
              self.raw_ptr(), IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &f,
              &linkage_name);
          if (iree_status_is_not_found(status)) {
            iree_status_ignore(status);
            break;
          }
          CheckApiStatus(status, "Error enumerating module");
          iree_string_view_t fname = iree_vm_function_name(&f);
          if (ordinal > 0) {
            repr.append(", ");
          }
          repr.append(fname.data, fname.size);
        }
        repr.append("]");
        repr.append(">");
        return repr;
      });
}

}  // namespace python
}  // namespace iree
