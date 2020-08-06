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

#include "bindings/python/pyiree/rt/function_abi.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "bindings/python/pyiree/common/status_utils.h"
#include "bindings/python/pyiree/rt/hal.h"
#include "bindings/python/pyiree/rt/vm.h"
#include "iree/base/api.h"
#include "iree/base/signature_mangle.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/list.h"
#include "iree/vm/ref.h"

namespace iree {
namespace python {

namespace {

// Python friendly entry-point for creating an instance from a list
// of attributes. This is not particularly efficient and is primarily
// for testing. Typically, this will be created directly from a function
// and the attribute introspection will happen internal to C++.
std::unique_ptr<FunctionAbi> PyCreateAbi(
    HalDevice& device, std::shared_ptr<HostTypeFactory> host_type_factory,
    std::vector<std::pair<std::string, std::string>> attrs) {
  auto lookup =
      [&attrs](absl::string_view key) -> absl::optional<absl::string_view> {
    for (const auto& kv : attrs) {
      if (kv.first == key) return kv.second;
    }
    return absl::nullopt;
  };
  return FunctionAbi::Create(device, std::move(host_type_factory), lookup);
}

VmVariantList PyRawPack(FunctionAbi* self,
                        absl::Span<const FunctionAbi::Description> descs,
                        py::sequence py_args, bool writable) {
  if (py_args.size() != descs.size()) {
    throw RaiseValueError("Mismatched pack arity");
  }

  VmVariantList f_args = VmVariantList::Create(py_args.size());
  absl::InlinedVector<py::handle, 8> local_py_args(py_args.begin(),
                                                   py_args.end());
  self->RawPack(descs, absl::MakeSpan(local_py_args), f_args, writable);
  return f_args;
}

VmVariantList PyAllocateResults(FunctionAbi* self, VmVariantList& f_args,
                                bool static_alloc) {
  auto f_results = VmVariantList::Create(self->raw_result_arity());
  if (static_alloc) {
    // For static dispatch, attempt to fully allocate and perform shape
    // inference.
    self->AllocateResults(absl::MakeConstSpan(self->raw_config().results),
                          f_args, f_results);
  }
  return f_results;
}

py::object PyRawUnpackResults(FunctionAbi* self, VmVariantList& f_args) {
  absl::InlinedVector<py::object, 4> py_results;
  py_results.resize(f_args.size());
  self->RawUnpack(absl::MakeConstSpan(self->raw_config().results), f_args,
                  absl::MakeSpan(py_results));
  py::tuple py_result_tuple(py_results.size());
  for (size_t i = 0, e = py_results.size(); i < e; ++i) {
    py_result_tuple[i] = std::move(py_results[i]);
  }
  return std::move(py_result_tuple);  // Without move, warns of copy.
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

pybind11::error_already_set RaiseBufferMismatchError(
    std::string message, py::handle obj,
    const RawSignatureParser::Description& desc) {
  message.append("For argument = ");
  auto arg_py_str = py::str(obj);
  auto arg_str = static_cast<std::string>(arg_py_str);
  message.append(arg_str);
  message.append(" (expected ");
  desc.ToString(message);
  message.append(")");
  return RaiseValueError(message.c_str());
}

// Verifies and maps the py buffer shape and layout to the bound argument.
// Returns false if not compatible.
void MapBufferAttrs(Py_buffer& py_view,
                    const RawSignatureParser::Description& desc,
                    absl::InlinedVector<int, 2>& dynamic_dims) {
  // Verify that rank matches.
  if (py_view.ndim != desc.dims.size()) {
    throw RaiseBufferMismatchError(
        absl::StrCat("Mismatched buffer rank (received: ", py_view.ndim,
                     ", expected: ", desc.dims.size(), "): "),
        py::handle(py_view.obj), desc);
  }

  // Verify that the item size matches.
  size_t f_item_size =
      AbiConstants::kScalarTypeSize[static_cast<int>(desc.buffer.scalar_type)];
  if (f_item_size != py_view.itemsize) {
    throw RaiseBufferMismatchError(
        absl::StrCat("Mismatched buffer item size (received: ",
                     py_view.itemsize, ", expected: ", f_item_size, "): "),
        py::handle(py_view.obj), desc);
  }

  // Note: The python buffer format does not map precisely to IREE's type
  // system, so the below is only advisory for where they do match. Otherwise,
  // it is basically a bitcast.
  const char* f_expected_format =
      kScalarTypePyFormat[static_cast<int>(desc.buffer.scalar_type)];
  if (f_expected_format != nullptr &&
      strcmp(f_expected_format, py_view.format) != 0) {
    throw RaiseBufferMismatchError(
        absl::StrCat("Mismatched buffer format (received: ", py_view.format,
                     ", expected: ", f_expected_format, "): "),
        py::handle(py_view.obj), desc);
  }

  // Verify shape, populating dynamic_dims while looping.
  for (size_t i = 0; i < py_view.ndim; ++i) {
    auto py_dim = py_view.shape[i];
    auto f_dim = desc.dims[i];
    if (f_dim < 0) {
      // Dynamic.
      dynamic_dims.push_back(py_dim);
    } else if (py_dim != f_dim) {
      // Mismatch.
      throw RaiseBufferMismatchError(
          absl::StrCat("Mismatched buffer dim (received: ", py_dim,
                       ", expected: ", f_dim, "): "),
          py::handle(py_view.obj), desc);
    }
  }
}

void PackScalar(const RawSignatureParser::Description& desc, py::handle py_arg,
                VmVariantList& f_args) {
  iree_vm_value value;
  value.type = IREE_VM_VALUE_TYPE_I32;
  switch (desc.scalar.type) {
    case AbiConstants::ScalarType::kUint8:
    case AbiConstants::ScalarType::kUint16:
    case AbiConstants::ScalarType::kUint32: {
      value.i32 = py_arg.cast<int32_t>();
      break;
    }
    case AbiConstants::ScalarType::kSint8:
    case AbiConstants::ScalarType::kSint16:
    case AbiConstants::ScalarType::kSint32: {
      value.i32 = py_arg.cast<int32_t>();
      break;
    }
    default:
      throw RaisePyError(PyExc_NotImplementedError, "Unsupported scalar type");
  }
  CheckApiStatus(iree_vm_list_push_value(f_args.raw_ptr(), &value),
                 "Could not pack scalar argument");
}

py::object UnpackScalar(const RawSignatureParser::Description& desc,
                        const iree_vm_variant_t& f_result) {
  switch (desc.scalar.type) {
    case AbiConstants::ScalarType::kUint8:
    case AbiConstants::ScalarType::kUint16:
    case AbiConstants::ScalarType::kUint32: {
      return py::int_(static_cast<uint32_t>(f_result.i32));
    }
    case AbiConstants::ScalarType::kSint8:
    case AbiConstants::ScalarType::kSint16:
    case AbiConstants::ScalarType::kSint32: {
      return py::int_(f_result.i32);
    }
    default:
      throw RaisePyError(PyExc_NotImplementedError, "Unsupported scalar type");
  }
}

}  // namespace

//------------------------------------------------------------------------------
// FunctionAbi
//------------------------------------------------------------------------------

std::string FunctionAbi::DebugString() const {
  RawSignatureParser p;
  auto s = p.FunctionSignatureToString(raw_config_.signature);
  if (!s) {
    return "<FunctionAbi NO_DEBUG_INFO>";
  }
  return absl::StrCat("<FunctionAbi ", *s, ">");
}

std::unique_ptr<FunctionAbi> FunctionAbi::Create(
    HalDevice& device, std::shared_ptr<HostTypeFactory> host_type_factory,
    AttributeLookup lookup) {
  auto abi =
      absl::make_unique<FunctionAbi>(device, std::move(host_type_factory));

  // Fetch key attributes for the raw ABI.
  auto raw_version = lookup("fv");
  auto raw_fsig_str = lookup("f");

  // Validation.
  if (!raw_fsig_str) {
    throw RaiseValueError("No raw abi reflection metadata for function");
  }
  if (!raw_version || *raw_version != "1") {
    throw RaiseValueError("Unsupported raw function ABI version");
  }

  // Parse signature.
  abi->raw_config().signature = std::string(*raw_fsig_str);
  RawSignatureParser raw_parser;
  raw_parser.VisitInputs(*raw_fsig_str,
                         [&abi](const RawSignatureParser::Description& d) {
                           abi->raw_config().inputs.push_back(d);
                         });
  raw_parser.VisitResults(*raw_fsig_str,
                          [&abi](const RawSignatureParser::Description& d) {
                            abi->raw_config().results.push_back(d);
                          });
  if (raw_parser.GetError()) {
    auto message = absl::StrCat(
        "Error parsing raw ABI signature: ", *raw_parser.GetError(), " ('",
        *raw_fsig_str, "')");
    throw RaiseValueError(message.c_str());
  }

  // TODO(laurenzo): Detect sip ABI and add a translation layer.
  return abi;
}

void FunctionAbi::RawPack(absl::Span<const Description> descs,
                          absl::Span<py::handle> py_args, VmVariantList& f_args,
                          bool writable) {
  if (descs.size() != py_args.size()) {
    throw RaiseValueError("Mismatched RawPack() input arity");
  }

  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer:
        PackBuffer(desc, py_args[i], f_args, writable);
        break;
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      case RawSignatureParser::Type::kScalar:
        PackScalar(desc, py_args[i], f_args);
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported argument type");
    }
  }
}

void FunctionAbi::RawUnpack(absl::Span<const Description> descs,
                            VmVariantList& f_results,
                            absl::Span<py::object> py_results) {
  if (descs.size() != f_results.size() || descs.size() != py_results.size()) {
    throw RaiseValueError("Mismatched RawUnpack() result arity");
  }
  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    iree_vm_variant_t f_result = iree_vm_variant_empty();
    iree_status_t status =
        iree_vm_list_get_variant(f_results.raw_ptr(), i, &f_result);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      throw RaiseValueError("Could not get result from list");
    }
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer: {
        iree_hal_buffer_view_t* buffer_view =
            iree_hal_buffer_view_deref(&f_result.ref);
        if (!buffer_view) {
          throw RaiseValueError(
              "Could not deref result buffer view (wrong type?)");
        }
        iree_hal_buffer* raw_buffer = iree_hal_buffer_view_buffer(buffer_view);
        if (!raw_buffer) {
          throw RaiseValueError("Could not deref result buffer (wrong type?)");
        }
        HalBuffer buffer = HalBuffer::RetainAndCreate(raw_buffer);

        // Extract dims from the buffer view.
        size_t rank = 0;
        absl::InlinedVector<int32_t, 6> dims(6);
        iree_status_t status = iree_hal_buffer_view_shape(
            buffer_view, dims.capacity(), dims.data(), &rank);
        if (iree_status_is_out_of_range(status)) {
          dims.resize(rank);
          status = iree_hal_buffer_view_shape(buffer_view, dims.capacity(),
                                              dims.data(), &rank);
        }
        CheckApiStatus(status, "Error extracting shape");
        dims.resize(rank);

        // Deal with int32_t != int (but require 32bits). Happens on some
        // embedded platforms.
        static_assert(sizeof(dims[0]) == sizeof(int),
                      "expected int to be 32 bits");
        py_results[i] = host_type_factory_->CreateImmediateNdarray(
            desc.buffer.scalar_type,
            absl::MakeConstSpan(reinterpret_cast<int*>(dims.data()),
                                dims.size()),
            std::move(buffer));
        break;
      }
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      case RawSignatureParser::Type::kScalar:
        py_results[i] = UnpackScalar(desc, f_result);
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported result type");
    }
  }
}

void FunctionAbi::AllocateResults(absl::Span<const Description> descs,
                                  VmVariantList& f_args,
                                  VmVariantList& f_results) {
  if (f_args.size() != raw_config().inputs.size()) {
    throw RaiseValueError("Mismatched AllocateResults() input arity");
  }

  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    iree_device_size_t alloc_size =
        AbiConstants::kScalarTypeSize[static_cast<int>(
            desc.buffer.scalar_type)];
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer: {
        absl::InlinedVector<int32_t, 5> dims;
        for (auto dim : desc.dims) {
          if (dim < 0) {
            // If there is a dynamic dim, fallback to completely func allocated
            // result. This is the worst case because it will force a
            // pipeline stall.
            // TODO(laurenzo): Invoke shape resolution function if available
            // to allocate full result.
            f_results.AppendNullRef();
          }
          alloc_size *= dim;
          dims.push_back(dim);
        }

        // Static cases are easy.
        iree_hal_buffer_t* raw_buffer;
        CheckApiStatus(iree_hal_allocator_allocate_buffer(
                           device_.allocator(),
                           static_cast<iree_hal_memory_type_t>(
                               IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                               IREE_HAL_MEMORY_TYPE_HOST_VISIBLE),
                           IREE_HAL_BUFFER_USAGE_ALL, alloc_size, &raw_buffer),
                       "Error allocating host visible buffer");
        auto element_type = static_cast<iree_hal_element_type_t>(
            kScalarTypeToHalElementType[static_cast<unsigned>(
                desc.scalar.type)]);
        iree_hal_buffer_view_t* buffer_view;
        CheckApiStatus(iree_hal_buffer_view_create(
                           raw_buffer, dims.data(), dims.size(), element_type,
                           iree_allocator_system(), &buffer_view),
                       "Error allocating buffer_view");
        iree_hal_buffer_release(raw_buffer);
        iree_vm_ref_t buffer_view_ref =
            iree_hal_buffer_view_move_ref(buffer_view);
        CheckApiStatus(
            iree_vm_list_push_ref_move(f_results.raw_ptr(), &buffer_view_ref),
            "Error moving buffer");
        break;
      }
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      case RawSignatureParser::Type::kScalar:
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported allocation argument type");
    }
  }
}

void FunctionAbi::PackBuffer(const RawSignatureParser::Description& desc,
                             py::handle py_arg, VmVariantList& f_args,
                             bool writable) {
  // Request a view of the buffer (use the raw python C API to avoid some
  // allocation and copying at the pybind level).
  Py_buffer py_view;
  // Note that only C-Contiguous ND-arrays are presently supported, so
  // only request that via PyBUF_ND. Long term, we should consult an
  // "oracle" in the runtime to determine the precise required format and
  // set flags accordingly (and fallback/copy on failure).
  int flags = PyBUF_FORMAT | PyBUF_ND;
  if (writable) {
    flags |= PyBUF_WRITABLE;
  }

  // Acquire the backing buffer and setup RAII release.
  if (PyObject_GetBuffer(py_arg.ptr(), &py_view, flags) != 0) {
    // The GetBuffer call is required to set an appropriate error.
    throw py::error_already_set();
  }
  PyBufferReleaser py_view_releaser(py_view);

  // Whether the py object needs to be retained with the argument.
  // Should be set to true if directly mapping, false if copied.
  bool depends_on_pyobject = false;

  // Verify compatibility.
  absl::InlinedVector<int, 2> dynamic_dims;
  MapBufferAttrs(py_view, desc, dynamic_dims);

  // Allocate a HalBuffer.
  // This is hard-coded to C-contiguous right now.
  // TODO(laurenzo): Expand to other layouts as needed.
  // TODO(laurenzo): Wrap and retain original buffer (depends_on_pyobject=true).
  iree_hal_buffer_t* raw_buffer;
  CheckApiStatus(iree_hal_allocator_allocate_buffer(
                     device_.allocator(),
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
  auto element_type = static_cast<iree_hal_element_type_t>(
      kScalarTypeToHalElementType[static_cast<unsigned>(desc.scalar.type)]);
  absl::InlinedVector<int, 5> dims(py_view.ndim);
  std::copy(py_view.shape, py_view.shape + py_view.ndim, dims.begin());
  iree_hal_buffer_view_t* buffer_view;
  CheckApiStatus(iree_hal_buffer_view_create(
                     raw_buffer, dims.data(), dims.size(), element_type,
                     iree_allocator_system(), &buffer_view),
                 "Error allocating buffer_view");
  iree_hal_buffer_release(raw_buffer);
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  CheckApiStatus(iree_vm_list_push_ref_move(f_args.raw_ptr(), &buffer_view_ref),
                 "Error moving buffer view");
}

void SetupFunctionAbiBindings(pybind11::module m) {
  py::class_<FunctionAbi, std::unique_ptr<FunctionAbi>>(m, "FunctionAbi")
      .def(py::init(&PyCreateAbi))
      .def("__repr__", &FunctionAbi::DebugString)
      .def_property_readonly("raw_input_arity", &FunctionAbi::raw_input_arity)
      .def_property_readonly("raw_result_arity", &FunctionAbi::raw_result_arity)
      .def("raw_pack_inputs",
           [](FunctionAbi* self, py::sequence py_args) {
             return PyRawPack(self,
                              absl::MakeConstSpan(self->raw_config().inputs),
                              py_args, false /* writable */);
           })
      .def("allocate_results", &PyAllocateResults, py::arg("f_results"),
           py::arg("static_alloc") = true)
      .def("raw_unpack_results", &PyRawUnpackResults);
}

}  // namespace python
}  // namespace iree
