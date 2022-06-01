// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./vm.h"

#include "./status_utils.h"
#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
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
  return VmModule::StealFromRawPtr(module);
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
  IREE_TRACE_SCOPE0("VmInstance::Create");
  iree_vm_instance_t* instance;
  auto status = iree_vm_instance_create(iree_allocator_system(), &instance);
  CheckApiStatus(status, "Error creating instance");
  return VmInstance::StealFromRawPtr(instance);
}

//------------------------------------------------------------------------------
// VmContext
//------------------------------------------------------------------------------

VmContext VmContext::Create(VmInstance* instance,
                            std::optional<std::vector<VmModule*>> modules) {
  IREE_TRACE_SCOPE0("VmContext::Create");
  iree_vm_context_t* context;
  if (!modules) {
    // Simple create with open allowed modules.
    auto status =
        iree_vm_context_create(instance->raw_ptr(), IREE_VM_CONTEXT_FLAG_NONE,
                               iree_allocator_system(), &context);
    CheckApiStatus(status, "Error creating vm context");
  } else {
    // Closed set of modules.
    std::vector<iree_vm_module_t*> module_handles;
    module_handles.resize(modules->size());
    for (size_t i = 0, e = module_handles.size(); i < e; ++i) {
      module_handles[i] = (*modules)[i]->raw_ptr();
    }
    auto status = iree_vm_context_create_with_modules(
        instance->raw_ptr(), IREE_VM_CONTEXT_FLAG_NONE, module_handles.data(),
        module_handles.size(), iree_allocator_system(), &context);
    CheckApiStatus(status, "Error creating vm context with modules");
  }

  IREE_CHECK(context);
  return VmContext::StealFromRawPtr(context);
}

void VmContext::RegisterModules(std::vector<VmModule*> modules) {
  std::vector<iree_vm_module_t*> module_handles;
  module_handles.resize(modules.size());
  for (size_t i = 0, e = module_handles.size(); i < e; ++i) {
    module_handles[i] = modules[i]->raw_ptr();
  }
  auto status = iree_vm_context_register_modules(raw_ptr(), &module_handles[0],
                                                 module_handles.size());
  CheckApiStatus(status, "Error registering modules");
}

void VmContext::Invoke(iree_vm_function_t f, VmVariantList& inputs,
                       VmVariantList& outputs) {
  iree_status_t status;
  {
    py::gil_scoped_release release;
    status = iree_vm_invoke(raw_ptr(), f, IREE_VM_INVOCATION_FLAG_NONE, nullptr,
                            inputs.raw_ptr(), outputs.raw_ptr(),
                            iree_allocator_system());
  }
  CheckApiStatus(status, "Error invoking function");
}

//------------------------------------------------------------------------------
// VmModule
//------------------------------------------------------------------------------

VmModule VmModule::FromFlatbufferBlob(py::object flatbuffer_blob_object) {
  IREE_TRACE_SCOPE0("VmModule::FromFlatbufferBlob");
  auto flatbuffer_blob = py::cast<py::buffer>(flatbuffer_blob_object);
  auto buffer_info = flatbuffer_blob.request();
  iree_vm_module_t* module;

  // Bridge to the C-based deallocator API.
  auto* raw_ptr = flatbuffer_blob.ptr();
  auto ctl_fn = +([](void* self, iree_allocator_command_t command,
                     const void* params, void** inout_ptr) {
    assert(command == IREE_ALLOCATOR_COMMAND_FREE);
    PyObject* object_ptr = static_cast<PyObject*>(*inout_ptr);
    Py_XDECREF(object_ptr);
    return iree_ok_status();
  });
  flatbuffer_blob.inc_ref();
  iree_allocator_t deallocator{/*self=*/NULL, /*ctl=*/ctl_fn};

  auto status = iree_vm_bytecode_module_create(
      {static_cast<const uint8_t*>(buffer_info.ptr),
       static_cast<iree_host_size_t>(buffer_info.size)},
      deallocator, iree_allocator_system(), &module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(deallocator, raw_ptr);
  }

  CheckApiStatus(status, "Error creating vm module from FlatBuffer");
  auto py_module = VmModule::StealFromRawPtr(module);
  py_module.stashed_flatbuffer_blob = flatbuffer_blob_object;
  return py_module;
}

std::optional<iree_vm_function_t> VmModule::LookupFunction(
    const std::string& name, iree_vm_function_linkage_t linkage) {
  iree_vm_function_t f;
  auto status = iree_vm_module_lookup_function_by_name(
      raw_ptr(), linkage, {name.data(), name.size()}, &f);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return std::nullopt;
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

void VmVariantList::PushBufferView(HalBufferView& buffer_view) {
  iree_vm_ref_t buffer_view_ref =
      iree_hal_buffer_view_retain_ref(buffer_view.raw_ptr());
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
      return GetAsBufferView(index);
    }
  }

  throw RaiseValueError("Unsupported VM to Python Type Conversion");
}

py::object VmVariantList::GetAsSerializedTraceValue(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant(raw_ptr(), index, &v),
                 "Could not access list element");
  if (iree_vm_type_def_is_value(&v.type)) {
    // Convert a value type.
    py::dict record;
    switch (v.type.value_type) {
      case IREE_VM_VALUE_TYPE_I8:
        record["i8"] = py::cast(v.i8);
        break;
      case IREE_VM_VALUE_TYPE_I16:
        record["i16"] = py::cast(v.i16);
        break;
      case IREE_VM_VALUE_TYPE_I32:
        record["i32"] = py::cast(v.i32);
        break;
      case IREE_VM_VALUE_TYPE_I64:
        record["i64"] = py::cast(v.i64);
        break;
      case IREE_VM_VALUE_TYPE_F32:
        record["f32"] = py::cast(v.f32);
        break;
      case IREE_VM_VALUE_TYPE_F64:
        record["f64"] = py::cast(v.f64);
        break;
      default:
        throw RaiseValueError("Unsupported VM value type conversion");
    }
    record["type"] = py::cast("value");
    return std::move(record);
  } else if (v.type.ref_type == IREE_VM_REF_TYPE_NULL) {
    py::dict record;
    record["type"] = "null";
    return std::move(record);
  } else if (iree_vm_type_def_is_ref(&v.type)) {
    // Convert reference type.
    if (iree_vm_list_isa(v.ref)) {
      py::dict record;
      record["type"] = "vm.list";
      py::list items;
      iree_vm_list_t* sub_list = NULL;
      CheckApiStatus(iree_vm_list_check_deref(v.ref, &sub_list),
                     "Could not deref list (wrong type?)");
      iree_vm_list_retain(sub_list);
      VmVariantList sub_list_object(sub_list);
      for (int i = 0, e = sub_list_object.size(); i < e; ++i) {
        items.append(sub_list_object.GetAsSerializedTraceValue(i));
      }
      record["items"] = std::move(items);
      return std::move(record);
    } else if (iree_hal_buffer_view_isa(v.ref)) {
      py::dict record;
      record["type"] = "hal.buffer_view";
      iree_hal_buffer_view_t* buffer_view = iree_hal_buffer_view_deref(v.ref);
      if (!buffer_view) {
        throw RaiseValueError(
            "Could not deref result buffer view (wrong type?)");
      }
      iree_hal_buffer_t* raw_buffer = iree_hal_buffer_view_buffer(buffer_view);
      if (!raw_buffer) {
        throw RaiseValueError("Could not deref result buffer (wrong type?)");
      }

      // Extract dims from the buffer view.
      size_t rank = 0;
      std::vector<iree_hal_dim_t> dims(6);
      iree_status_t status = iree_hal_buffer_view_shape(
          buffer_view, dims.capacity(), dims.data(), &rank);
      if (iree_status_is_out_of_range(status)) {
        dims.resize(rank);
        status = iree_hal_buffer_view_shape(buffer_view, dims.capacity(),
                                            dims.data(), &rank);
      }
      CheckApiStatus(status, "Error extracting shape");
      dims.resize(rank);
      record["shape"] = py::cast(std::move(dims));

      // Element type.
      iree_hal_element_type_t element_type =
          iree_hal_buffer_view_element_type(buffer_view);
      // TODO: Would be nice to output as hex.
      record["element_type"] = element_type;

      // Map memory.
      iree_device_size_t byte_length = iree_hal_buffer_byte_length(raw_buffer);
      iree_hal_buffer_mapping_t mapped_memory = {{0}};
      CheckApiStatus(iree_hal_buffer_map_range(
                         raw_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                         IREE_HAL_MEMORY_ACCESS_READ, 0 /* element_offset */,
                         byte_length, &mapped_memory),
                     "Could not map memory");
      record["contents"] =
          py::bytes(reinterpret_cast<const char*>(mapped_memory.contents.data),
                    mapped_memory.contents.data_length);
      iree_hal_buffer_unmap_range(&mapped_memory);

      return std::move(record);
    }
  }

  throw RaiseValueError("Unsupported VM to Python Type Conversion");
}

py::object VmVariantList::GetAsBufferView(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant(raw_ptr(), index, &v),
                 "Could not access list element");
  iree_hal_buffer_view_t* buffer_view = iree_hal_buffer_view_deref(v.ref);
  if (!buffer_view) {
    throw RaiseValueError("Could not deref result buffer view (wrong type?)");
  }
  return py::cast(HalBufferView::BorrowFromRawPtr(buffer_view),
                  py::return_value_policy::move);
}

namespace {

static std::string ToHexString(const uint8_t* data, size_t length) {
  static constexpr char kHexChars[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                       '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string s(length * 2, ' ');
  for (size_t i = 0; i < length; ++i) {
    s[2 * i + 0] = kHexChars[(data[i] & 0xF0) >> 4];
    s[2 * i + 1] = kHexChars[(data[i] & 0x0F) >> 0];
  }
  return s;
}
static std::string ToHexString(uint32_t value) {
  return ToHexString((const uint8_t*)&value, sizeof(value));
}

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
      // Convert a value type to a string.
      switch (variant.type.value_type) {
        case IREE_VM_VALUE_TYPE_I8: {
          out += std::to_string(variant.i8);
          break;
        }
        case IREE_VM_VALUE_TYPE_I16: {
          out += std::to_string(variant.i16);
          break;
        }
        case IREE_VM_VALUE_TYPE_I32: {
          out += std::to_string(variant.i32);
          break;
        }
        case IREE_VM_VALUE_TYPE_I64: {
          out += std::to_string(variant.i64);
          break;
        }
        case IREE_VM_VALUE_TYPE_F32: {
          out += std::to_string(variant.f32);
          break;
        }
        case IREE_VM_VALUE_TYPE_F64: {
          out += std::to_string(variant.f64);
          break;
        }
        default:
          throw RaiseValueError("Unsupported VM value type to string");
      }
    } else if (iree_vm_variant_is_ref(variant)) {
      // Pretty print a subset of ABI impacting known types.
      if (iree_hal_buffer_isa(variant.ref)) {
        auto* hal_buffer = iree_hal_buffer_deref(variant.ref);
        assert(hal_buffer);
        out += std::string("HalBuffer(") +
               std::to_string(iree_hal_buffer_byte_length(hal_buffer)) + ")";
      } else if (iree_hal_buffer_view_isa(variant.ref)) {
        auto hal_bv = iree_hal_buffer_view_deref(variant.ref);
        out += "HalBufferView(";
        std::vector<iree_hal_dim_t> shape(
            iree_hal_buffer_view_shape_rank(hal_bv));
        iree_hal_buffer_view_shape(hal_bv, shape.size(), shape.data(), nullptr);
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i > 0) out += 'x';
          out += std::to_string(shape[i]);
        }
        out += ":0x" +
               ToHexString(static_cast<uint32_t>(
                   iree_hal_buffer_view_element_type(hal_bv))) +
               ")";
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
        out += "Unknown(" + std::to_string(variant.type.ref_type) + ")";
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
  std::string s =
      std::string("<VmVariantList(") + std::to_string(size()) + "): [";
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

  // Built-in module creation.
  m.def("create_hal_module", &CreateHalModule);

  py::enum_<enum iree_vm_function_linkage_e>(m, "Linkage")
      .value("INTERNAL", IREE_VM_FUNCTION_LINKAGE_INTERNAL)
      .value("IMPORT", IREE_VM_FUNCTION_LINKAGE_IMPORT)
      .value("IMPORT_OPTIONAL", IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL)
      .value("EXPORT", IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .export_values();

  // Mutation and inspection of the variant list is mostly opaque to python.
  py::class_<VmVariantList>(m, "VmVariantList")
      .def(py::init(&VmVariantList::Create))
      .def_property_readonly("size", &VmVariantList::size)
      .def("__len__", &VmVariantList::size)
      .def("get_as_buffer_view", &VmVariantList::GetAsBufferView)
      .def("get_as_list", &VmVariantList::GetAsList)
      .def("get_variant", &VmVariantList::GetVariant)
      .def("get_serialized_trace_value",
           &VmVariantList::GetAsSerializedTraceValue)
      .def("push_float", &VmVariantList::PushFloat)
      .def("push_int", &VmVariantList::PushInt)
      .def("push_list", &VmVariantList::PushList)
      .def("push_buffer_view", &VmVariantList::PushBufferView)
      .def("__repr__", &VmVariantList::DebugString);

  py::class_<iree_vm_function_t>(m, "VmFunction")
      .def_readonly("linkage", &iree_vm_function_t::linkage)
      .def_readonly("ordinal", &iree_vm_function_t::ordinal)
      .def_property_readonly("name",
                             [](iree_vm_function_t& self) {
                               iree_string_view_t name =
                                   iree_vm_function_name(&self);
                               return py::str(name.data, name.size);
                             })
      .def_property_readonly("module_name",
                             [](iree_vm_function_t& self) {
                               iree_string_view_t name =
                                   iree_vm_module_name(self.module);
                               return py::str(name.data, name.size);
                             })
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
           py::arg("modules") = std::optional<std::vector<VmModule*>>())
      .def("register_modules", &VmContext::RegisterModules)
      .def_property_readonly("context_id", &VmContext::context_id)
      .def("invoke", &VmContext::Invoke);

  py::class_<VmModule>(m, "VmModule")
      .def_static("from_flatbuffer", &VmModule::FromFlatbufferBlob)
      .def_property_readonly("name", &VmModule::name)
      .def("lookup_function", &VmModule::LookupFunction, py::arg("name"),
           py::arg("linkage") = IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .def_property_readonly(
          "stashed_flatbuffer_blob",
          [](VmModule& self) { return self.get_stashed_flatbuffer_blob(); })
      .def_property_readonly(
          "function_names",
          [](VmModule& self) {
            py::list names;
            iree_vm_module_signature_t sig =
                iree_vm_module_signature(self.raw_ptr());
            for (size_t ordinal = 0; ordinal < sig.export_function_count;
                 ++ordinal) {
              iree_vm_function_t f;
              auto status = iree_vm_module_lookup_function_by_ordinal(
                  self.raw_ptr(), IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &f);
              if (iree_status_is_not_found(status)) {
                iree_status_ignore(status);
                break;
              }
              CheckApiStatus(status, "Error enumerating module");
              iree_string_view_t fname = iree_vm_function_name(&f);
              py::str name(fname.data, fname.size);
              names.append(name);
            }
            return names;
          })
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
          auto status = iree_vm_module_lookup_function_by_ordinal(
              self.raw_ptr(), IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &f);
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
