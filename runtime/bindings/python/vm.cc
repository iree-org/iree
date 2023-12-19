// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./vm.h"

#include <ios>
#include <sstream>
#include <unordered_set>

#include "./buffer_interop.h"
#include "./status_utils.h"
#include "iree/base/api.h"

// TODO: We shouldn't need the HAL API but it is used for direct printing
// summaries of HAL objects in lists. We should have a better way of doing this
// dynamically vs hard depending on a type switch here.
#include "iree/modules/hal/module.h"
#include "iree/tooling/modules/resolver.h"
#include "iree/vm/api.h"

using namespace nanobind::literals;

namespace iree {
namespace python {

namespace {

static const char kFromBufferDocstring[] =
    R"(Creates a Vmmodule from a Python buffer.

This is intended as a quick and dirty way to instantiate a VmModule from
a binary blob. It will implicitly make a copy if alignment is not sufficient.

It is recommended to use one of the other construction methods for maximum
determinism and efficiency:

* `mmap` : To memory map from a file.
* `wrap_buffer` : To directly wrap a Python buffer that is known to be
  aligned properly.
* `copy_buffer` : To always make a copy of a Python buffer such that it is
  aligned properly.

This was historically called `from_flatbuffer`. It is recommended that new
code use `flat_buffer`.

Args:
  instance: A VmInstance.
  buffer: An object implementing the Python buffer protocol. Typically a
    bytes, bytearray, memoryview, etc.
  warn_if_copy: Raises a warning if alignment is not sufficient to use the
    buffer directly, resulting in a copy. Defaults to True.
)";

static const char kCopyBufferDocstring[] =
    R"(Creates a VmModule by making a copy of a Python buffer.

Args:
  instance: A VmInstance.
  buffer: An object implementing the Python buffer protocol. Typically a
    bytes, bytearray, memoryview, etc.
)";

static const char kWrapBufferDocstring[] =
    R"(Creates a VmModule by directly using the backing memory of a Python buffer.

Args:
  instance: A VmInstance.
  buffer: An object implementing the Python buffer protocol. Typically a
    bytes, bytearray, memoryview, etc.
  destroy_callback: A no-argument callback that is invoked when the backing
    buffer is no longer in use.
  close_buffer: Whether to call the `close` method on the `buffer` (prior to
    invoking `destroy_callback`). Defaults to False.

Raises:
  ValueError if alignment is not satisfied.
)";

static const char kMMapDocstring[] =
    R"(Create a VmModule by mmap'ing a file.

When backed by a file, this is generally the most effective way to create a
VmModule. Especially for large modules, this will result in the fewest
copies and the most effective use of the system cache across invocations.

Note that mmap behavior differs between Posix and Windows. Whereas the former
will allow the backing file to be open before an mmap call and deleted
immediately after, Windows generally allows neither. For compatibility,
make sure that the backing file is not open for writing before calling this
method and that if it needs to be deleted when done, that is done in a
`destroy_callback`.

Args:
  instance: A VmInstance.
  filepath: Path to the file on the file system.
  destroy_callback: A no-argument callback that is invoked when the backing
    buffer is no longer in use.
)";

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
  for (iree_host_size_t i = 0;; ++i) {
    iree_string_pair_t attr;
    auto status = iree_vm_function_get_attr(f, i, &attr);
    if (iree_status_is_out_of_range(status)) {
      iree_status_ignore(status);
      break;
    }
    CheckApiStatus(status, "Error getting reflection attr");
    py::str key_str(attr.key.data, attr.key.size);
    py::str value_str(attr.value.data, attr.value.size);
    attrs[std::move(key_str)] = std::move(value_str);
  }
  return attrs;
}

}  // namespace

//------------------------------------------------------------------------------
// VmInstance
//------------------------------------------------------------------------------

VmInstance VmInstance::Create() {
  IREE_TRACE_SCOPE_NAMED("VmInstance::Create");

  iree_vm_instance_t* instance = NULL;
  auto status = iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance);
  CheckApiStatus(status, "Error creating instance");

  // The python bindings assume the HAL is always available for use.
  // We register the types here so modules can be loaded using the HAL types
  // in any order.
  CheckApiStatus(iree_hal_module_register_all_types(instance),
                 "registering HAL types");

  return VmInstance::StealFromRawPtr(instance);
}

//------------------------------------------------------------------------------
// VmContext
//------------------------------------------------------------------------------

VmContext VmContext::Create(VmInstance* instance,
                            std::optional<std::vector<VmModule*>>& modules) {
  IREE_TRACE_SCOPE_NAMED("VmContext::Create");
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
        instance->raw_ptr(), IREE_VM_CONTEXT_FLAG_NONE, module_handles.size(),
        module_handles.data(), iree_allocator_system(), &context);
    CheckApiStatus(status, "Error creating vm context with modules");
  }

  IREE_ASSERT(context);
  return VmContext::StealFromRawPtr(context);
}

void VmContext::RegisterModules(std::vector<VmModule*> modules) {
  std::vector<iree_vm_module_t*> module_handles;
  module_handles.resize(modules.size());
  for (size_t i = 0, e = module_handles.size(); i < e; ++i) {
    module_handles[i] = modules[i]->raw_ptr();
  }
  auto status = iree_vm_context_register_modules(
      raw_ptr(), module_handles.size(), &module_handles[0]);
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

VmModule VmModule::ResolveModuleDependency(VmInstance* instance,
                                           const std::string& name,
                                           uint32_t minimum_version) {
  IREE_TRACE_SCOPE_NAMED("VmModule::ResolveModuleDependency");
  iree_vm_module_t* module = nullptr;

  iree_vm_module_dependency_t dependency = {
      iree_make_cstring_view(name.c_str()), minimum_version,
      IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED};

  auto status = iree_tooling_resolve_module_dependency(
      instance->raw_ptr(), &dependency, iree_allocator_system(), &module);

  assert(module != nullptr);

  CheckApiStatus(status, "Error resolving module dependency");
  auto py_module = VmModule::StealFromRawPtr(module);
  return py_module;
}

VmModule VmModule::MMap(VmInstance* instance, std::string filepath,
                        py::object destroy_callback) {
  IREE_TRACE_SCOPE_NAMED("VmModule::MMap");
  auto mmap_module = py::module_::import_("mmap");
  auto open_func = py::module_::import_("io").attr("open");
  auto file_obj = open_func(filepath, "r+b");
  // The signature of mmap is different on Windows vs others. On others,
  // we use explicit flags and protection attributes for better control,
  // triggering off of the presence of the MAP_SHARED flag constant (which
  // is not present on Windows).
  py::object mapped_file;
  if (py::hasattr(mmap_module, "MAP_SHARED")) {
    // Posix mmap signature.
    auto flags = py::cast<int64_t>(mmap_module.attr("MAP_SHARED"));
    // MAP_POPULATE isn't available on all versions/platforms.
    if (py::hasattr(mmap_module, "MAP_POPULATE")) {
      flags |= py::cast<int64_t>(mmap_module.attr("MAP_POPULATE"));
    }
    auto prot = py::cast<int64_t>(mmap_module.attr("PROT_READ"));
    mapped_file = mmap_module.attr("mmap")(file_obj.attr("fileno")(), 0,
                                           "flags"_a = flags, "prot"_a = prot);
  } else {
    // Windows mmap signature.
    mapped_file =
        mmap_module.attr("mmap")(file_obj.attr("fileno")(), 0,
                                 "access"_a = mmap_module.attr("ACCESS_READ"));
  }
  // Backing file can be closed after a successful mmap call.
  file_obj.attr("close")();

  // MADV_RANDOM is not available on Windows (and possibly others?).
  if (py::hasattr(mmap_module, "MADV_RANDOM")) {
    mapped_file.attr("madvise")(mmap_module.attr("MADV_RANDOM"));
  }
  return WrapBuffer(instance, std::move(mapped_file),
                    std::move(destroy_callback),
                    /*close_buffer=*/true);
}

VmModule VmModule::WrapBuffer(VmInstance* instance, py::object buffer_obj,
                              py::object destroy_callback, bool close_buffer) {
  IREE_TRACE_SCOPE_NAMED("VmModule::FromAlignedMemory");
  // State object that is retained for the life of the module.
  // It is responsible for keeping the backing resources alive and
  // holding the user-level destroy callback.
  // Note that the original buffer_obj is not captured explicitly but
  // is available as part of the Py_buffer underlying the PyBufferRequest.
  // Aside from being more efficient, avoiding redundant capture removes
  // destruction race potential.
  struct BufferState {
    BufferState(py::object buffer_obj, py::object destroy_callback,
                bool close_buffer)
        : buffer_info(buffer_obj, PyBUF_SIMPLE),
          destroy_callback(std::move(destroy_callback)),
          close_buffer(close_buffer) {}
    PyBufferRequest buffer_info;
    py::object destroy_callback;
    bool close_buffer;

    py::handle get_buffer() { return py::handle(buffer_info.view().obj); }
  };
  BufferState* state =
      new BufferState(buffer_obj, destroy_callback, close_buffer);
  PyBufferRequest& buffer_info = state->buffer_info;
  if (!iree_host_size_has_alignment((uintptr_t)buffer_info.view().buf,
                                    IREE_HAL_HEAP_BUFFER_ALIGNMENT)) {
    std::stringstream err;
    err << "VmModule.from_aligned_memory received an unaligned buffer. ";
    err << "Got 0x" << (void*)buffer_info.view().buf << ", expected alignment ";
    err << IREE_HAL_HEAP_BUFFER_ALIGNMENT;
    throw std::invalid_argument(err.str());
  }

  iree_vm_module_t* module = nullptr;
  auto ctl_fn = +([](void* self, iree_allocator_command_t command,
                     const void* params, void** inout_ptr) {
    py::gil_scoped_acquire gil;
    assert(command == IREE_ALLOCATOR_COMMAND_FREE);
    try {
      // Destruction sequencing is tricky. We must have released the
      // PyBufferRequest before calling close, so we first get what we
      // need out of the state into local variables, then delete the state
      // (releasing the PyBufferRequest), then closing and issuing the
      // destroy callback. Getting the order wrong will result in an
      // unrecoverable exception indicating the the buffer cannot be closed
      // with outstanding mappings.
      BufferState* state = static_cast<BufferState*>(self);
      py::object destroy_callback = std::move(state->destroy_callback);
      py::object buffer_to_close;
      if (state->close_buffer) {
        buffer_to_close = py::borrow(state->get_buffer());
      }
      delete state;

      if (buffer_to_close) {
        buffer_to_close.attr("close")();
      }

      if (!destroy_callback.is_none()) {
        destroy_callback();
      }
    } catch (std::exception& e) {
      // There are many situations where deallocation exceptions can be
      // swallowed, so carp loudly. This is almost always a critical issue
      // that needs to be visible.
      fprintf(
          stderr,
          "error: exception raised while deallocating storage for an "
          "iree.runtime.VmModule. This is unrecoverable and likely indicates a "
          "serious problem, minimally resulting in memory leaks: %s",
          e.what());
      return iree_make_status(
          IREE_STATUS_UNKNOWN,
          "exception raised while deallocating storage for an "
          "iree.runtime.VmModule. This is unrecoverable and likely indicates a "
          "serious problem, minimally resulting in memory leaks: %s",
          e.what());
    }
    return iree_ok_status();
  });
  iree_allocator_t deallocator{/*self=*/state, /*ctl=*/ctl_fn};

  auto status = iree_vm_bytecode_module_create(
      instance->raw_ptr(),
      {static_cast<const uint8_t*>(buffer_info.view().buf),
       static_cast<iree_host_size_t>(buffer_info.view().len)},
      deallocator, iree_allocator_system(), &module);
  if (!iree_status_is_ok(status)) {
    delete state;
  }

  CheckApiStatus(status, "Error creating vm module from aligned memory");
  auto py_module = VmModule::StealFromRawPtr(module);
  // Stash a reference to the flatbuffer at the Python instance level. This
  // is exposed to the tracing API, allowing it to get at the backing contents.
  py_module.stashed_flatbuffer_blob = buffer_obj;
  return py_module;
}

VmModule VmModule::CopyBuffer(VmInstance* instance, py::object buffer_obj) {
  IREE_TRACE_SCOPE_NAMED("VmModule::CopyBuffer");
  auto alignment =
      py::cast<uintptr_t>(py::module_::import_("mmap").attr("PAGESIZE"));
  auto bytearray_ctor = py::module_::import_("builtins").attr("bytearray");
  PyBufferRequest src_buffer_info(buffer_obj, PyBUF_SIMPLE);
  auto src_buffer_size = src_buffer_info.view().len;

  // Need to allocate an extra page because there is no control at the Python
  // level for the alignment it may have.
  auto dst_buffer = bytearray_ctor(src_buffer_size + alignment);
  PyBufferRequest dst_buffer_info(dst_buffer, PyBUF_SIMPLE);
  void* dst_aligned =
      (void*)iree_host_align((uintptr_t)dst_buffer_info.view().buf, alignment);
  uintptr_t dst_offset =
      (uintptr_t)dst_aligned - (uintptr_t)dst_buffer_info.view().buf;

  // Now create a memoryview over the unaligned bytearray and slice into that
  // to get the aligned Python buffer.
  auto dst_slice =
      py::slice(py::cast(dst_offset), py::cast(dst_offset + src_buffer_size),
                py::cast(1));

  py::object dst_view = py::steal<py::object>(
      PyMemoryView_GetContiguous(dst_buffer.ptr(), PyBUF_READ, 'C'));
  py::object dst_view_aligned = dst_view[dst_slice];

  // If any of the indexing math was wrong, Python exceptions will be raised
  // above, so this is implicitly guarding the memcpy if it is done last.
  std::memcpy(dst_aligned, src_buffer_info.view().buf, src_buffer_size);
  return WrapBuffer(instance, std::move(dst_view_aligned),
                    /*destroy_callback=*/py::none(),
                    /*close_buffer=*/false);
}

VmModule VmModule::FromBuffer(VmInstance* instance, py::object buffer_obj,
                              bool warn_if_copy) {
  IREE_TRACE_SCOPE_NAMED("VmModule::FromBuffer");
  PyBufferRequest buffer_info(buffer_obj, PyBUF_SIMPLE);

  if (iree_host_size_has_alignment((uintptr_t)buffer_info.view().buf,
                                   IREE_HAL_HEAP_BUFFER_ALIGNMENT)) {
    return WrapBuffer(instance, std::move(buffer_obj),
                      /*destroy_callback=*/py::none(), /*close_buffer=*/false);
  } else {
    if (warn_if_copy) {
      py::module_::import_("warnings")
          .attr("warn")(
              "Making copy of unaligned VmModule buffer. It is recommended to "
              "make this deterministic by calling `copy_buffer` to always make "
              "a copy or `mmap` to efficiently load from a file. This warning "
              "can be silenced by adding `warn_if_copy=False` to "
              "`from_buffer`");
    }
    return CopyBuffer(instance, std::move(buffer_obj));
  }
}

std::optional<iree_vm_function_t> VmModule::LookupFunction(
    const std::string& name, iree_vm_function_linkage_t linkage) {
  iree_vm_function_t f;
  auto status = iree_vm_module_lookup_function_by_name(
      raw_ptr(), linkage,
      {name.data(), static_cast<iree_host_size_t>(name.size())}, &f);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return std::nullopt;
  }
  CheckApiStatus(status, "Error looking up function");
  return f;
}

//------------------------------------------------------------------------------
// VmRef
//------------------------------------------------------------------------------

const char* const VmRef::kRefAttr = "__iree_vm_ref__";
const char* const VmRef::kCastAttr = "__iree_vm_cast__";
const char* const VmRef::kTypeAttr = "__iree_vm_type__";

py::object VmRef::Deref(py::object ref_object_class, bool optional) {
  py::object casted = ref_object_class.attr(kCastAttr)(this);
  if (!optional && casted.is_none()) {
    throw py::type_error("Cannot dereference to specific type");
  }
  return casted;
}

bool VmRef::IsInstance(py::object ref_object_class) {
  auto type = py::cast<iree_vm_ref_type_t>(ref_object_class.attr(kTypeAttr)());
  return type == ref_.type;
}

std::string VmRef::ToString() {
  if (!ref_.ptr) {
    return "<VmRef NULL>";
  }
  iree_string_view_t type_name = iree_vm_ref_type_name(ref_.type);
  std::stringstream ss;
  ss << "<VmRef ";
  ss.write(type_name.data, type_name.size);
  ss << " at " << std::hex << "0x" << reinterpret_cast<uintptr_t>(ref_.ptr)
     << ">";
  return ss.str();
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

void VmVariantList::PushRef(py::handle ref_or_object) {
  py::object py_ref = ref_or_object.attr(VmRef::kRefAttr);
  VmRef& ref = py::cast<VmRef&>(py_ref);
  CheckApiStatus(iree_vm_list_push_ref_retain(raw_ptr(), &ref.ref()),
                 "Failed to push ref");
}

py::object VmVariantList::GetAsList(int index) {
  iree_vm_ref_t ref = {0};
  CheckApiStatus(iree_vm_list_get_ref_assign(raw_ptr(), index, &ref),
                 "Could not access list element");
  iree_vm_list_t* sub_list = NULL;
  CheckApiStatus(iree_vm_list_check_deref(ref, &sub_list),
                 "Could not deref list (wrong type?)");
  iree_vm_list_retain(sub_list);
  return py::cast(VmVariantList::StealFromRawPtr(sub_list));
}

py::object VmVariantList::GetVariant(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant_assign(raw_ptr(), index, &v),
                 "Could not access list element");
  if (iree_vm_variant_is_empty(v)) {
    return py::none();
  } else if (iree_vm_variant_is_value(v)) {
    // Convert a value type.
    switch (iree_vm_type_def_as_value(v.type)) {
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
  } else if (iree_vm_variant_is_ref(v)) {
    VmRef ref;
    iree_vm_ref_retain(&v.ref, &ref.ref());
    return py::cast(ref, py::rv_policy::move);
  }

  throw RaiseValueError("Unsupported VM to Python Type Conversion");
}

py::object VmVariantList::GetAsSerializedTraceValue(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant_assign(raw_ptr(), index, &v),
                 "Could not access list element");
  if (iree_vm_variant_is_empty(v)) {
    py::dict record;
    record["type"] = "null";
    return std::move(record);
  } else if (iree_vm_variant_is_value(v)) {
    // Convert a value type.
    py::dict record;
    switch (iree_vm_type_def_as_value(v.type)) {
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
  } else if (iree_vm_variant_is_ref(v)) {
    // Convert reference type.
    if (iree_vm_list_isa(v.ref)) {
      py::dict record;
      record["type"] = "vm.list";
      py::list items;
      iree_vm_list_t* sub_list = NULL;
      CheckApiStatus(iree_vm_list_check_deref(v.ref, &sub_list),
                     "Could not deref list (wrong type?)");
      iree_vm_list_retain(sub_list);
      VmVariantList sub_list_object = VmVariantList::StealFromRawPtr(sub_list);
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
      iree_host_size_t rank = 0;
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
      char element_type_str[64] = {0};
      iree_host_size_t element_type_length = 0;
      CheckApiStatus(
          iree_hal_format_element_type(element_type, sizeof(element_type_str),
                                       element_type_str, &element_type_length),
          "Formatting element type");
      record["element_type"] =
          std::string(element_type_str, element_type_length);

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

py::object VmVariantList::GetAsRef(int index) {
  iree_vm_variant_t v = iree_vm_variant_empty();
  CheckApiStatus(iree_vm_list_get_variant_assign(raw_ptr(), index, &v),
                 "Could not access list element");
  if (!iree_vm_variant_is_ref(v)) {
    throw std::invalid_argument("list element is not a ref");
  }
  VmRef ref;
  iree_vm_ref_retain(&v.ref, &ref.ref());
  return py::cast(ref, py::rv_policy::move);
}

py::object VmVariantList::GetAsObject(int index, py::object clazz) {
  return clazz.attr(VmRef::kCastAttr)(GetAsRef(index));
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
    iree_status_t status = iree_vm_list_get_variant_assign(list, i, &variant);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      out.append("Error");
      continue;
    }
    if (i > 0) out.append(", ");

    if (iree_vm_variant_is_value(variant)) {
      // Convert a value type to a string.
      switch (iree_vm_type_def_as_value(variant.type)) {
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
      } else if (iree_hal_fence_isa(variant.ref)) {
        out.append("fence(");
        auto* hal_fence = iree_hal_fence_deref(variant.ref);
        iree_host_size_t timepoint_count =
            iree_hal_fence_timepoint_count(hal_fence);
        out.append(std::to_string(timepoint_count) + ")");
      } else {
        out += "Unknown(" +
               std::to_string(iree_vm_type_def_as_ref(variant.type)) + ")";
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

void SetupVmBindings(nanobind::module_ m) {
  py::enum_<enum iree_vm_function_linkage_e>(m, "Linkage")
      .value("INTERNAL", IREE_VM_FUNCTION_LINKAGE_INTERNAL)
      .value("IMPORT", IREE_VM_FUNCTION_LINKAGE_IMPORT)
      .value("IMPORT_OPTIONAL", IREE_VM_FUNCTION_LINKAGE_IMPORT_OPTIONAL)
      .value("EXPORT", IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .export_values();

  auto vm_buffer = py::class_<VmBuffer>(m, "VmBuffer");
  VmRef::BindRefProtocol(vm_buffer, iree_vm_buffer_type,
                         iree_vm_buffer_retain_ref, iree_vm_buffer_deref,
                         iree_vm_buffer_isa);
  // Implement the buffer protocol with low-level API.
  {
    static PyBufferProcs buffer_procs = {
        // It is not legal to raise exceptions from these callbacks.
        +[](PyObject* raw_self, Py_buffer* view, int flags) -> int {
          // Cast must succeed due to invariants.
          auto self = py::cast<VmBuffer*>(py::handle(raw_self));
          if (view == NULL) {
            PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
            return -1;
          }

          Py_INCREF(raw_self);
          view->obj = raw_self;
          view->buf = self->raw_ptr()->data.data;
          view->len = self->raw_ptr()->data.data_length;
          view->readonly =
              !(self->raw_ptr()->access & IREE_VM_BUFFER_ACCESS_MUTABLE);
          view->itemsize = 1;
          view->format = (char*)"B";  // Byte
          view->ndim = 1;
          view->shape = nullptr;
          view->strides = nullptr;
          view->suboffsets = nullptr;
          view->internal = nullptr;
          return 0;
        },
        +[](PyObject* self_obj, Py_buffer* view) -> void {

        },
    };
    auto heap_type = reinterpret_cast<PyHeapTypeObject*>(vm_buffer.ptr());
    assert(heap_type->ht_type.tp_flags & Py_TPFLAGS_HEAPTYPE &&
           "must be heap type");
    heap_type->as_buffer = buffer_procs;
  }

  vm_buffer
      .def(
          "__init__",
          [](VmBuffer* self, iree_host_size_t length,
             iree_host_size_t alignment, bool is_mutable) {
            iree_vm_buffer_access_t access = 0;
            if (is_mutable) {
              access |= IREE_VM_BUFFER_ACCESS_MUTABLE;
            }
            iree_vm_buffer_t* raw_buffer;
            CheckApiStatus(
                iree_vm_buffer_create(access, length, alignment,
                                      iree_allocator_system(), &raw_buffer),
                "Error creating buffer");

            new (self) VmBuffer();
            *self = VmBuffer::StealFromRawPtr(raw_buffer);
          },
          py::arg("length"), py::arg("alignment") = 0,
          py::arg("mutable") = true)
      .def("__repr__", [](VmBuffer& self) {
        std::stringstream ss;
        ss << "<VmBuffer size " << self.raw_ptr()->data.data_length << " at 0x"
           << std::hex << reinterpret_cast<uintptr_t>(self.raw_ptr()->data.data)
           << ">";
        return ss.str();
      });

  // Mutation and inspection of the variant list is mostly opaque to python.
  auto vm_list = py::class_<VmVariantList>(m, "VmVariantList");
  VmRef::BindRefProtocol(vm_list, iree_vm_list_type, iree_vm_list_retain_ref,
                         iree_vm_list_deref, iree_vm_list_isa);
  vm_list
      // User Methods.
      .def(
          "__init__",
          [](VmVariantList* self, iree_host_size_t capacity) {
            new (self) VmVariantList();
            *self = VmVariantList::Create(capacity);
          },
          py::arg("capacity"))
      .def_prop_ro("size", &VmVariantList::size)
      .def("__len__", &VmVariantList::size)
      .def("get_as_ref", &VmVariantList::GetAsRef)
      .def("get_as_object", &VmVariantList::GetAsObject)
      .def("get_as_list", &VmVariantList::GetAsList)
      .def("get_variant", &VmVariantList::GetVariant)
      .def("get_serialized_trace_value",
           &VmVariantList::GetAsSerializedTraceValue)
      .def("push_float", &VmVariantList::PushFloat)
      .def("push_int", &VmVariantList::PushInt)
      .def("push_list", &VmVariantList::PushList)
      .def("push_ref", &VmVariantList::PushRef)
      .def("__repr__", &VmVariantList::DebugString);

  py::class_<iree_vm_function_t>(m, "VmFunction")
      .def_ro("linkage", &iree_vm_function_t::linkage)
      .def_ro("ordinal", &iree_vm_function_t::ordinal)
      .def_prop_ro("name",
                   [](iree_vm_function_t& self) {
                     iree_string_view_t name = iree_vm_function_name(&self);
                     return py::str(name.data, name.size);
                   })
      .def_prop_ro("module_name",
                   [](iree_vm_function_t& self) {
                     iree_string_view_t name = iree_vm_module_name(self.module);
                     return py::str(name.data, name.size);
                   })
      .def_prop_ro("reflection",
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

  py::class_<VmInstance>(m, "VmInstance").def("__init__", [](VmInstance* self) {
    new (self) VmInstance();
    *self = VmInstance::Create();
  });
  py::class_<VmContext>(m, "VmContext")
      .def(
          "__init__",
          [](VmContext* self, VmInstance* instance,
             std::optional<std::vector<VmModule*>> modules) {
            new (self) VmContext();
            *self = VmContext::Create(instance, modules);
          },
          py::arg("instance"),
          py::arg("modules") = std::optional<std::vector<VmModule*>>())
      .def("register_modules", &VmContext::RegisterModules)
      .def_prop_ro("context_id", &VmContext::context_id)
      .def("invoke", &VmContext::Invoke);

  py::class_<VmModule>(m, "VmModule")
      .def_static("resolve_module_dependency",
                  &VmModule::ResolveModuleDependency)
      .def_static("from_flatbuffer", &VmModule::FromBuffer, py::arg("instance"),
                  py::arg("buffer"), py::arg("warn_if_copy") = true,
                  kFromBufferDocstring)
      .def_static("from_buffer", &VmModule::FromBuffer, py::arg("instance"),
                  py::arg("buffer"), py::arg("warn_if_copy") = true,
                  kFromBufferDocstring)
      .def_static("copy_buffer", &VmModule::CopyBuffer, py::arg("instance"),
                  py::arg("buffer"), kCopyBufferDocstring)
      .def_static("wrap_buffer", &VmModule::WrapBuffer, py::arg("instance"),
                  py::arg("buffer"), py::arg("destroy_callback") = py::none(),
                  py::arg("close_buffer") = false, kWrapBufferDocstring)
      .def_static("mmap", &VmModule::MMap, py::arg("instance"),
                  py::arg("filepath"), py::arg("destroy_callback") = py::none(),
                  kMMapDocstring)
      .def_prop_ro("name", &VmModule::name)
      .def_prop_ro("version",
                   [](VmModule& self) {
                     iree_vm_module_signature_t sig =
                         iree_vm_module_signature(self.raw_ptr());
                     return sig.version;
                   })
      .def("lookup_function", &VmModule::LookupFunction, py::arg("name"),
           py::arg("linkage") = IREE_VM_FUNCTION_LINKAGE_EXPORT)
      .def_prop_ro(
          "stashed_flatbuffer_blob",
          [](VmModule& self) { return self.get_stashed_flatbuffer_blob(); })
      .def_prop_ro("function_names",
                   [](VmModule& self) {
                     py::list names;
                     iree_vm_module_signature_t sig =
                         iree_vm_module_signature(self.raw_ptr());
                     for (size_t ordinal = 0;
                          ordinal < sig.export_function_count; ++ordinal) {
                       iree_vm_function_t f;
                       auto status = iree_vm_module_lookup_function_by_ordinal(
                           self.raw_ptr(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
                           ordinal, &f);
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

  py::class_<VmRef>(m, "VmRef")
      .def("isinstance", &VmRef::IsInstance)
      .def("deref", &VmRef::Deref, py::arg("value"),
           py::arg("optional") = false)
      .def("__repr__", &VmRef::ToString)
      .def_prop_ro(VmRef::kRefAttr, [](py::object self) { return self; })
      .def("__eq__",
           [](VmRef& self, VmRef& other) {
             return self.ref().ptr == other.ref().ptr;
           })
      .def("__eq__", [](VmRef& self, py::object& other) { return false; });
}

}  // namespace python
}  // namespace iree
