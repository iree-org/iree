// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./py_module.h"

#include <string_view>

#include "./vm.h"

namespace iree::python {

// Low level class for constructing a native VM module from Python. This
// class is mutable while the module is being setup and will typically
// produce a module instance when ready to be used.
//
// This class has a complicated life-cycle and can be in one of several
// states:
//   UNINITIALZED: Prior to calling Create(). Mutable.
//   INITIALIZED: After calling Create() and prior to the returned reference
//     being released. Immutable.
//   DESTROYED: After the reference from Create() is released. Nothing
//     more can be done with the instance but it is still live until the
//     Python reference to it is released.
class PyModuleInterface {
 public:
  PyModuleInterface(std::string module_name, py::object ctor)
      : module_name_(std::move(module_name)), ctor_(std::move(ctor)) {
    CheckApiStatus(iree_vm_module_initialize(&interface_, this),
                   "Failed to initialize vm_module");
    interface_.destroy = &PyModuleInterface::ModuleDestroy;
    interface_.name = &PyModuleInterface::ModuleName;
    interface_.signature = &PyModuleInterface::ModuleSignature;
    interface_.enumerate_dependencies =
        &PyModuleInterface::ModuleEnumerateDependencies;
    interface_.get_function = &PyModuleInterface::ModuleGetFunction;
    interface_.lookup_function = &PyModuleInterface::ModuleLookupFunction;
    interface_.alloc_state = &PyModuleInterface::ModuleAllocState;
    interface_.free_state = &PyModuleInterface::ModuleFreeState;
    interface_.resolve_import = &PyModuleInterface::ModuleResolveImport;
    interface_.notify = &PyModuleInterface::ModuleNotify;
    interface_.begin_call = &PyModuleInterface::ModuleBeginCall;
  }
  PyModuleInterface(const PyModuleInterface&) = delete;
  ~PyModuleInterface() = default;

  static PyModuleInterface* AsSelf(void* vself) {
    return static_cast<PyModuleInterface*>(vself);
  }

  static void ModuleDestroy(void* vself) {
    auto self = AsSelf(vself);
    py::gil_scoped_acquire acquire;
    self->retained_self_ref_ = {};
  }

  static iree_string_view_t ModuleName(void* vself) {
    auto self = AsSelf(vself);
    return {self->module_name_.data(), self->module_name_.size()};
  }

  static iree_vm_module_signature_t ModuleSignature(void* vself) {
    auto self = AsSelf(vself);
    iree_vm_module_signature_t signature = {0};
    signature.version = self->descriptor_.version;
    signature.attr_count = 0;
    signature.import_function_count = self->imports_.size();
    signature.export_function_count = self->exports_.size();
    signature.internal_function_count = 0;
    return signature;
  }

  static iree_status_t ModuleEnumerateDependencies(
      void* vself, iree_vm_module_dependency_callback_t callback,
      void* user_data) {
    // TODO(laurenzo): python support for declaring dependencies on the module.
    return iree_ok_status();
  }

  static iree_status_t ModuleGetFunction(
      void* vself, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal,
      iree_vm_function_t* out_function, iree_string_view_t* out_name,
      iree_vm_function_signature_t* out_signature) {
    auto self = AsSelf(vself);
    if (IREE_LIKELY(linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT)) {
      if (IREE_LIKELY(ordinal < self->export_functions_.size())) {
        std::unique_ptr<PyFunction>& f = self->export_functions_[ordinal];
        if (IREE_LIKELY(out_function)) {
          out_function->linkage = linkage;
          out_function->module = &self->interface_;
          out_function->ordinal = ordinal;
        }
        if (IREE_LIKELY(out_name)) {
          *out_name = {f->name.data(), f->name.size()};
        }
        if (IREE_LIKELY(out_signature)) {
          out_signature->calling_convention = {f->cconv.data(),
                                               f->cconv.size()};
        }
        return iree_ok_status();
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND);
  }

  static iree_status_t ModuleLookupFunction(
      void* vself, iree_vm_function_linkage_t linkage, iree_string_view_t name,
      const iree_vm_function_signature_t* expected_signature,
      iree_vm_function_t* out_function) {
    auto self = AsSelf(vself);
    std::string_view name_cpp(name.data, name.size);
    if (linkage == IREE_VM_FUNCTION_LINKAGE_EXPORT) {
      auto found_it = self->export_name_to_ordinals_.find(name_cpp);
      if (found_it != self->export_name_to_ordinals_.end()) {
        out_function->linkage = linkage;
        out_function->module = &self->interface_;
        out_function->ordinal = found_it->second;
        return iree_ok_status();
      }
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND, "function %.*s not exported",
                            (int)name.size, name.data);
  }

  static iree_status_t ModuleAllocState(
      void* vself, iree_allocator_t allocator,
      iree_vm_module_state_t** out_module_state) {
    auto self = AsSelf(vself);
    *out_module_state = nullptr;
    py::gil_scoped_acquire acquire;
    try {
      py::object py_state = self->ctor_(self->retained_self_ref_);
      // Steal the reference and use the raw PyObject* as the state.
      // This will be released in ModuleFreeState.
      *out_module_state =
          reinterpret_cast<iree_vm_module_state_t*>(py_state.release().ptr());
      return iree_ok_status();
    } catch (std::exception& e) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "Exception in call to PyModule constructor: %s",
                              e.what());
    }
  }

  static void ModuleFreeState(void* vself,
                              iree_vm_module_state_t* module_state) {
    py::gil_scoped_acquire acquire;
    // Release the reference stolen in ModuleAllocState.
    auto retained_handle =
        py::handle(reinterpret_cast<PyObject*>(module_state));
    retained_handle.dec_ref();
  }

  static iree_status_t ModuleResolveImport(
      void* vself, iree_vm_module_state_t* module_state,
      iree_host_size_t ordinal, const iree_vm_function_t* function,
      const iree_vm_function_signature_t* signature) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Python API does not support imports");
  }

  static iree_status_t ModuleNotify(void* vself,
                                    iree_vm_module_state_t* module_state,
                                    iree_vm_signal_t signal) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "ModuleNotify not implemented");
  }

  static iree_status_t ModuleBeginCall(void* vself, iree_vm_stack_t* stack,
                                       iree_vm_function_call_t call) {
    auto self = AsSelf(vself);
    if (IREE_UNLIKELY(call.function.ordinal >=
                      self->export_functions_.size())) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function ordinal out of bounds: 0 < %u < %zu",
                              call.function.ordinal,
                              self->export_functions_.size());
    }

    auto& f = self->export_functions_[call.function.ordinal];
    iree_host_size_t frame_size = 0;
    iree_vm_stack_frame_t* callee_frame = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_stack_function_enter(
        stack, &call.function, IREE_VM_STACK_FRAME_NATIVE, frame_size,
        /*frame_cleanup_fn=*/nullptr, &callee_frame));
    auto state_object =
        py::handle(reinterpret_cast<PyObject*>(callee_frame->module_state));

    try {
      IREE_RETURN_IF_ERROR(self->Invoke(*f, state_object, stack, call));
    } catch (std::exception& e) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "Exception raised from Python module: %s",
                              e.what());
    }

    return iree_vm_stack_function_leave(stack);
  }

  std::string ToString() {
    std::string s("<iree.runtime.PyModuleInterface '");
    s.append(module_name_);
    s.append("'");
    if (initialized_) {
      if (retained_self_ref_) {
        s.append(" initialized");
      } else {
        s.append(" destroyed");
      }
    }
    s.append(">");
    return s;
  }

  bool initialized() { return initialized_; }

  bool destroyed() { return initialized_ && !retained_self_ref_; }

  void AssertMutable() {
    if (initialized_) {
      throw std::runtime_error("Attempt to mutate a frozen PyModuleInterface");
    }
  }

  void ExportFunction(std::string name, std::string cconv,
                      py::object callable) {
    // Make sure not already defined.
    if (export_name_to_ordinals_.count(name)) {
      std::string msg("PyModule function already defined: ");
      msg.append(name);
      throw std::invalid_argument(std::move(msg));
    }

    // Heap allocate the backing PyFunction so we can reference its pointers.
    size_t ordinal = exports_.size();
    auto py_function = std::make_unique<PyFunction>(
        std::move(name), std::move(cconv), std::move(callable));
    exports_.push_back({});
    iree_vm_native_export_descriptor_t& d = exports_.back();
    d.local_name = {py_function->name.data(), py_function->name.size()};
    d.calling_convention = {py_function->cconv.data(),
                            py_function->cconv.size()};
    d.attr_count = 0;
    d.attrs = nullptr;
    std::string& alloced_name = py_function->name;
    CheckApiStatus(py_function->ParseCconv(), "Unparseable calling convention");

    // Transfer the PyFunction to its vector now that we are done touching it.
    export_functions_.push_back(std::move(py_function));
    export_name_to_ordinals_.insert(
        std::make_pair(std::string_view(alloced_name), ordinal));
  }

  // Initializes the internal data structures such that GetInterface() will be
  // valid. After this call, the interface is "live" and this instance will only
  // be deleted when its refcnt goes to 0, which will call ModuleDestroy and
  // release our Python side reference to this.
  void Initialize() {
    AssertMutable();
    initialized_ = true;
    memset(&descriptor_, 0, sizeof(descriptor_));
    descriptor_.name = {module_name_.data(), module_name_.size()};
    descriptor_.version = version_;
    descriptor_.attr_count = attrs_.size();
    descriptor_.attrs = attrs_.empty() ? nullptr : attrs_.data();
    descriptor_.import_count = imports_.size();
    descriptor_.imports = imports_.empty() ? nullptr : imports_.data();
    descriptor_.export_count = exports_.size();
    descriptor_.exports = exports_.empty() ? nullptr : exports_.data();
    descriptor_.function_count = functions_.size();
    descriptor_.functions = functions_.empty() ? nullptr : functions_.data();
    retained_self_ref_ = py::cast(this);
  }

  // Creates the live Python VmModule reference. This can only be called once.
  VmModule Create() {
    Initialize();
    return VmModule::StealFromRawPtr(&interface_);
  }

 private:
  struct PyFunction {
    std::string name;
    std::string cconv;
    py::object callable;

    // Initialized by ParseCconv.
    iree_string_view_t cconv_arguments;
    iree_string_view_t cconv_results;

    PyFunction(std::string name, std::string cconv, py::object callable)
        : name(std::move(name)),
          cconv(std::move(cconv)),
          callable(std::move(callable)) {}

    iree_status_t ParseCconv() {
      iree_vm_function_signature_t signature;
      memset(&signature, 0, sizeof(signature));
      signature.calling_convention = {cconv.data(), cconv.size()};
      IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
          &signature, &cconv_arguments, &cconv_results));

      if (iree_vm_function_call_is_variadic_cconv(cconv_arguments) ||
          iree_vm_function_call_is_variadic_cconv(cconv_results)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "PyModules do not yet support variadic arguments/results");
      }

      return iree_ok_status();
    }
  };

  iree_status_t Invoke(PyFunction& f, py::handle state_object,
                       iree_vm_stack_t* stack, iree_vm_function_call_t call) {
    py::gil_scoped_acquire acquire;
    uint8_t* packed_arguments = call.arguments.data;
    iree_host_size_t packed_arguments_required_size;
    // TODO: Is this validation needed or do we assume it from up-stack?
    IREE_RETURN_IF_ERROR(iree_vm_function_call_compute_cconv_fragment_size(
        f.cconv_arguments, /*segment_size_list=*/nullptr,
        &packed_arguments_required_size));
    if (IREE_UNLIKELY(packed_arguments_required_size !=
                      call.arguments.data_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "mismatched packed argument size: actual=%zu, required=%zu",
          call.arguments.data_length, packed_arguments_required_size);
    }

    // Unpack arguments.
    py::list arguments;
    for (iree_host_size_t i = 0; i < f.cconv_arguments.size; ++i) {
      switch (f.cconv_arguments.data[i]) {
        case IREE_VM_CCONV_TYPE_VOID:
          break;
        case IREE_VM_CCONV_TYPE_I32:
          arguments.append(
              py::cast(*reinterpret_cast<int32_t*>(packed_arguments)));
          packed_arguments += sizeof(int32_t);
          break;
        case IREE_VM_CCONV_TYPE_F32:
          arguments.append(
              py::cast(*reinterpret_cast<float*>(packed_arguments)));
          packed_arguments += sizeof(float);
          break;
        case IREE_VM_CCONV_TYPE_I64:
          arguments.append(
              py::cast(*reinterpret_cast<int64_t*>(packed_arguments)));
          packed_arguments += sizeof(int64_t);
          break;
        case IREE_VM_CCONV_TYPE_F64:
          arguments.append(
              py::cast(*reinterpret_cast<double*>(packed_arguments)));
          packed_arguments += sizeof(double);
          break;
        case IREE_VM_CCONV_TYPE_REF: {
          iree_vm_ref_t ref =
              *reinterpret_cast<iree_vm_ref_t*>(packed_arguments);
          // Since the Python level VmRef can escape, it needs its own ref
          // count.
          VmRef py_ref;
          iree_vm_ref_retain(&ref, &py_ref.ref());
          arguments.append(py::cast(py_ref, py::return_value_policy::move));
          packed_arguments += sizeof(iree_vm_ref_t);
          break;
        }
        // TODO: Variadic segments.
        default:
          return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported cconv type %c",
                                  f.cconv_arguments.data[i]);
      }
    }

    auto results = f.callable(state_object, *arguments);

    // Pack results.
    if (f.cconv_results.size == 0) {
      return iree_ok_status();
    }
    uint8_t* packed_results = call.results.data;
    bool unary_result = f.cconv_results.size == 1;
    auto pack_result = [&](py::object& value,
                           char cconv_type) -> iree_status_t {
      switch (cconv_type) {
        case IREE_VM_CCONV_TYPE_VOID:
          break;
        case IREE_VM_CCONV_TYPE_I32:
          *reinterpret_cast<int32_t*>(packed_results) =
              py::cast<int32_t>(value);
          packed_results += sizeof(int32_t);
          break;
        case IREE_VM_CCONV_TYPE_F32:
          *reinterpret_cast<float*>(packed_results) = py::cast<float>(value);
          packed_results += sizeof(float);
          break;
        case IREE_VM_CCONV_TYPE_I64:
          *reinterpret_cast<int64_t*>(packed_results) =
              py::cast<int64_t>(value);
          packed_results += sizeof(int64_t);
          break;
        case IREE_VM_CCONV_TYPE_F64:
          *reinterpret_cast<double*>(packed_results) = py::cast<double>(value);
          packed_results += sizeof(double);
          break;
        case IREE_VM_CCONV_TYPE_REF: {
          iree_vm_ref_t* result_ref =
              reinterpret_cast<iree_vm_ref_t*>(packed_results);
          if (value.is_none()) {
            return iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "expected ref returned from Python function but got None");
          }
          VmRef* py_ref = py::cast<VmRef*>(value);
          iree_vm_ref_retain(&py_ref->ref(), result_ref);
          packed_results += sizeof(iree_vm_ref_t);
          break;
        }
        // TODO: Refs (need a generic Python ref wrapper).
        // TODO: Variadic segments.
        default:
          return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported cconv type %c", cconv_type);
      }
      return iree_ok_status();
    };

    if (unary_result) {
      return pack_result(results, f.cconv_results.data[0]);
    } else {
      py::sequence results_seq = py::cast<py::sequence>(results);
      int result_index = 0;
      for (iree_host_size_t i = 0; i < f.cconv_results.size; ++i) {
        py::object next_result = results_seq[result_index++];
        IREE_RETURN_IF_ERROR(pack_result(next_result, f.cconv_results.data[i]));
      }
      return iree_ok_status();
    }
  }

  // Descriptor state is built up when mutable and then will be populated
  // on the descriptor when frozen.
  std::string module_name_;
  uint32_t version_;
  py::object ctor_;
  std::vector<iree_string_pair_t> attrs_;
  std::vector<iree_vm_native_import_descriptor_t> imports_;
  std::vector<iree_vm_native_export_descriptor_t> exports_;
  std::vector<std::unique_ptr<PyFunction>> export_functions_;
  std::vector<iree_vm_native_function_ptr_t> functions_;

  // Map of names to ordinals.
  std::unordered_map<std::string_view, int> export_name_to_ordinals_;

  // Once the builder is frozen, the descriptor will be valid.
  iree_vm_module_t interface_;
  iree_vm_native_module_descriptor_t descriptor_;

  // Read-only and descriptor populated when frozen.
  bool initialized_ = false;
  py::object retained_self_ref_;
};

void SetupPyModuleBindings(py::module& m) {
  py::class_<PyModuleInterface>(m, "PyModuleInterface")
      .def(py::init<std::string, py::object>(), py::arg("module_name"),
           py::arg("ctor"))
      .def("__str__", &PyModuleInterface::ToString)
      .def_property_readonly("initialized", &PyModuleInterface::initialized)
      .def_property_readonly("destroyed", &PyModuleInterface::destroyed)
      .def("create", &PyModuleInterface::Create)
      .def("export", &PyModuleInterface::ExportFunction, py::arg("name"),
           py::arg("cconv"), py::arg("callable"));
}

}  // namespace iree::python
