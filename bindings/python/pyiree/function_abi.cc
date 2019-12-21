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

#include "bindings/python/pyiree/function_abi.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "bindings/python/pyiree/hal.h"
#include "bindings/python/pyiree/status_utils.h"
#include "iree/base/api.h"
#include "iree/base/signature_mangle.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm2/ref.h"
#include "iree/vm2/variant_list.h"

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

std::unique_ptr<FunctionArgVariantList> PyRawPack(
    FunctionAbi* self, absl::Span<const FunctionAbi::Description> descs,
    py::sequence py_args, bool writable) {
  if (py_args.size() != descs.size()) {
    throw RaiseValueError("Mismatched pack arity");
  }
  auto f_list = absl::make_unique<FunctionArgVariantList>();
  f_list->contents().resize(py_args.size());
  absl::InlinedVector<py::handle, 8> local_py_args(py_args.begin(),
                                                   py_args.end());
  self->RawPack(descs, absl::MakeSpan(local_py_args),
                absl::MakeSpan(f_list->contents()), writable);
  return f_list;
}

std::unique_ptr<FunctionArgVariantList> PyAllocateResults(
    FunctionAbi* self, FunctionArgVariantList* f_args) {
  auto f_results = absl::make_unique<FunctionArgVariantList>();
  f_results->contents().resize(self->raw_result_arity());
  self->AllocateResults(absl::MakeConstSpan(self->raw_config().results),
                        absl::MakeConstSpan(f_args->contents()),
                        absl::MakeSpan(f_results->contents()));
  return f_results;
}

py::object PyRawUnpackResults(FunctionAbi* self,
                              FunctionArgVariantList* f_args) {
  absl::InlinedVector<py::object, 4> py_results;
  py_results.resize(f_args->contents().size());
  self->RawUnpack(absl::MakeConstSpan(self->raw_config().results),
                  absl::MakeSpan(f_args->contents()),
                  absl::MakeSpan(py_results));
  py::tuple py_result_tuple(py_results.size());
  for (size_t i = 0, e = py_results.size(); i < e; ++i) {
    py_result_tuple[i] = std::move(py_results[i]);
  }
  return py_result_tuple;
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
                    BoundHalBufferFunctionArg& bound_arg) {
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
      bound_arg.dynamic_dims.push_back(py_dim);
    } else if (py_dim != f_dim) {
      // Mismatch.
      throw RaiseBufferMismatchError(
          absl::StrCat("Mismatched buffer dim (received: ", py_dim,
                       ", expected: ", f_dim, "): "),
          py::handle(py_view.obj), desc);
    }
  }
}

std::string FunctionArgVariantListRepr(FunctionArgVariantList* self) {
  std::string s;
  struct Visitor {
    Visitor(std::string& s) : s(s) {}
    void operator()(std::nullptr_t) { s.append("None"); }
    void operator()(const BoundHalBufferFunctionArg& arg) {
      absl::StrAppend(&s, "HalBuffer(", arg.buffer.byte_length());
      if (!arg.dynamic_dims.empty()) {
        absl::StrAppend(&s, ", dynamic_dims=[");
        for (size_t i = 0, e = arg.dynamic_dims.size(); i < e; ++i) {
          if (i > 0) s.append(",");
          absl::StrAppend(&s, arg.dynamic_dims[i]);
        }
        absl::StrAppend(&s, "]");
      }
      if (arg.dependent_pyobject) {
        absl::StrAppend(&s, ", wraps=");
        std::string wraps_s = py::str(arg.dependent_pyobject);
        s.append(wraps_s);
      }
      absl::StrAppend(&s, ")");
    }
    std::string& s;
  };
  Visitor v(s);

  absl::StrAppend(&s, "<FunctionArgVariantList(", self->contents().size(),
                  "): [");
  for (size_t i = 0, e = self->contents().size(); i < e; ++i) {
    if (i > 0) s.append(", ");
    const auto& arg = self->contents()[i];
    absl::visit(v, arg);
  }
  absl::StrAppend(&s, "]>");

  return s;
}

}  // namespace

//------------------------------------------------------------------------------
// FunctionArgVariantList
//------------------------------------------------------------------------------

VmVariantList FunctionArgVariantList::ToVmVariantList() {
  struct Visitor {
    VmVariantList list;
    void operator()(std::nullptr_t) {
      CheckApiStatus(iree_vm_variant_list_append_null_ref(list.raw_ptr()),
                     "Error appending to variant list");
    }
    void operator()(BoundHalBufferFunctionArg& arg) {
      if (arg.dependent_pyobject) {
        throw RaiseValueError("Dependent buffer object not yet supported");
      }
      iree_vm_ref_t buffer_ref =
          iree_hal_buffer_retain_ref(arg.buffer.raw_ptr());
      CheckApiStatus(
          iree_vm_variant_list_append_ref_move(list.raw_ptr(), &buffer_ref),
          "Error moving buffer");
    }
  };

  Visitor visitor{VmVariantList::Create(contents_.size())};
  for (auto& arg_variant : contents_) {
    absl::visit(visitor, arg_variant);
  }

  return std::move(visitor.list);
}

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
                          absl::Span<py::handle> py_args,
                          absl::Span<FunctionArgVariant> f_args,
                          bool writable) {
  if (descs.size() != py_args.size() || descs.size() != f_args.size()) {
    throw RaiseValueError("Mismatched RawPack() input arity");
  }

  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer:
        PackBuffer(desc, py_args[i], f_args[i], writable);
        break;
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported argument type");
    }
  }
}

void FunctionAbi::RawUnpack(absl::Span<const Description> descs,
                            absl::Span<FunctionArgVariant> f_results,
                            absl::Span<py::object> py_results) {
  if (descs.size() != f_results.size() || descs.size() != py_results.size()) {
    throw RaiseValueError("Mismatched RawUnpack() result arity");
  }
  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer: {
        auto& bound_arg = absl::get<BoundHalBufferFunctionArg>(f_results[i]);
        absl::InlinedVector<int, 7> backing_full_dims;
        absl::Span<const int> dims;
        if (bound_arg.dynamic_dims.empty()) {
          // No dynamic dims. Just use the desc dims.
          dims = absl::MakeSpan(desc.dims);
        } else {
          // Splice the dims back together.
          size_t j = 0;
          for (auto dim : desc.dims) {
            if (dim >= 0) {
              backing_full_dims.push_back(dim);
            } else if (j < bound_arg.dynamic_dims.size()) {
              backing_full_dims.push_back(bound_arg.dynamic_dims[j++]);
            } else {
              throw RaiseValueError("Mismatched static and dynamic dims");
            }
          }
          dims = absl::MakeConstSpan(backing_full_dims);
        }
        py_results[i] = host_type_factory_->CreateImmediateNdarray(
            desc.buffer.scalar_type, absl::MakeSpan(desc.dims),
            std::move(bound_arg.buffer));
        // Already moved the buffer out. Clear the rest.
        f_results[i] = nullptr;
        break;
      }
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported argument type");
    }
  }
}

void FunctionAbi::AllocateResults(absl::Span<const Description> descs,
                                  absl::Span<const FunctionArgVariant> f_args,
                                  absl::Span<FunctionArgVariant> f_results) {
  if (descs.size() != f_results.size()) {
    throw RaiseValueError("Mismatched AllocateResults() result arity");
  }
  if (f_args.size() != raw_config().inputs.size()) {
    throw RaiseValueError("Mismatched AllocatResults() input arity");
  }

  for (size_t i = 0, e = descs.size(); i < e; ++i) {
    const Description& desc = descs[i];
    iree_device_size_t alloc_size =
        AbiConstants::kScalarTypeSize[static_cast<int>(
            desc.buffer.scalar_type)];
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer: {
        BoundHalBufferFunctionArg bound_result;
        for (auto dim : desc.dims) {
          if (dim < 0) {
            bound_result.dynamic_dims.push_back(dim);
            // TODO(laurenzo): Should accumulate dynamic dim slices into an
            // invocation of a result allocator function and then use it to
            // invoke.
            // TODO(laurenzo): Should fallback to completely unknown result
            // thunk.
            throw RaisePyError(
                PyExc_NotImplementedError,
                "AllocateResult() does not yet support dynamic dims");
          }
          alloc_size *= dim;
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
        bound_result.buffer = HalBuffer::CreateRetained(raw_buffer);
        assert(i < f_results.size());
        f_results[i] = std::move(bound_result);
        break;
      }
      case RawSignatureParser::Type::kRefObject:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Ref objects not yet supported");
        break;
      default:
        throw RaisePyError(PyExc_NotImplementedError,
                           "Unsupported argument type");
    }
  }
}

void FunctionAbi::PackBuffer(const RawSignatureParser::Description& desc,
                             py::handle py_arg, FunctionArgVariant& f_arg,
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

  auto& bound_arg = f_arg.emplace<BoundHalBufferFunctionArg>();
  // Whether the py object needs to be retained with the argument.
  // Should be set to true if directly mapping, false if copied.
  bool depends_on_pyobject = false;

  // Verify compatibility.
  MapBufferAttrs(py_view, desc, bound_arg);

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
  bound_arg.buffer = HalBuffer::CreateRetained(raw_buffer);
  CheckApiStatus(iree_hal_buffer_write_data(bound_arg.buffer.raw_ptr(), 0,
                                            py_view.buf, py_view.len),
                 "Error writing to input buffer");

  // Only capture the reference to the exporting object (incrementing it)
  // once guaranteed successful.
  if (depends_on_pyobject) {
    bound_arg.dependent_pyobject = py::object(py::handle(py_view.obj), false);
  }
}

void SetupFunctionAbiBindings(pybind11::module m) {
  m.def("create", &PyCreateAbi);
  py::class_<FunctionAbi, std::unique_ptr<FunctionAbi>>(m, "FunctionAbi")
      .def("__repr__", &FunctionAbi::DebugString)
      .def_property_readonly("raw_input_arity", &FunctionAbi::raw_input_arity)
      .def_property_readonly("raw_result_arity", &FunctionAbi::raw_result_arity)
      .def("raw_pack_inputs",
           [](FunctionAbi* self, py::sequence py_args) {
             return PyRawPack(self,
                              absl::MakeConstSpan(self->raw_config().inputs),
                              py_args, false /* writable */);
           })
      .def("allocate_results", &PyAllocateResults)
      .def("raw_unpack_results", &PyRawUnpackResults);

  // Note that FunctionArgVariantList is meant to be opaque to normal users,
  // but we provide some printing and other introspection to aid testing.
  py::class_<FunctionArgVariantList, std::unique_ptr<FunctionArgVariantList>>(
      m, "FunctionArgVariantList")
      .def_property_readonly(
          "size",
          [](FunctionArgVariantList* self) { return self->contents().size(); })
      .def("to_vm_variant_list", &FunctionArgVariantList::ToVmVariantList)
      .def("__repr__", &FunctionArgVariantListRepr);
}

}  // namespace python
}  // namespace iree
