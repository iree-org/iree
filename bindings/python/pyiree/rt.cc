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

#include "bindings/python/pyiree/rt.h"

#include "base/api.h"
#include "bindings/python/pyiree/status_utils.h"
#include "hal/api.h"

namespace iree {
namespace python {

HalBufferView RtContext::WrapPyBufferForInput(py::buffer py_buffer) {
  auto py_buffer_info = py_buffer.request(false /* writable */);
  if (py_buffer_info.ndim > IREE_SHAPE_MAX_RANK || py_buffer_info.ndim < 0) {
    RaiseValueError("Unsupported buffer rank");
  }
  if (py_buffer_info.size < 0) {
    RaiseValueError("Illegal buffer size");
  }

  // For the moment, allocate a device visible buffer of equivalent size and
  // copy into it.
  // TODO(laurenzo): Once sequencer is in place, switch to HeapBuffer, wrap
  // and retain the original buffer.
  iree_host_size_t byte_size = py_buffer_info.size * py_buffer_info.itemsize;
  HalBuffer buffer =
      AllocateDeviceVisible(byte_size, IREE_HAL_BUFFER_USAGE_CONSTANT |
                                           IREE_HAL_BUFFER_USAGE_TRANSFER |
                                           IREE_HAL_BUFFER_USAGE_DISPATCH);
  CheckApiStatus(iree_hal_buffer_write_data(buffer.raw_ptr(), 0,
                                            py_buffer_info.ptr, byte_size),
                 "Error writing to input buffer");

  // Create the buffer view.
  // TODO(laurenzo): This does no validation on dtype and only cares if the
  // elementsize matches. Figure out where to enforce actual dtype.
  iree_shape_t shape;
  shape.rank = py_buffer_info.ndim;

  // Verify strides are row-major.
  // TODO(laurenzo): Test this with rank>1.
  for (int i = 1; i < shape.rank; ++i) {
    if ((py_buffer_info.strides[i - 1] * py_buffer_info.itemsize) !=
        py_buffer_info.shape[i]) {
      RaiseValueError("Expected row-major layout");
    }
  }
  if (!py_buffer_info.strides.empty()) {
    if (py_buffer_info.strides.back() != 1) {
      RaiseValueError("Expected row-major layout");
    }
  }

  // Populate shape.
  for (int i = 0; i < shape.rank; ++i) {
    ssize_t dim = py_buffer_info.shape[i];
    if (dim < 0) {
      RaiseValueError("Unsupported negative dim");
    }
    shape.dims[i] = dim;
  }

  iree_hal_buffer_view_t* bv;
  CheckApiStatus(iree_hal_buffer_view_create(buffer.raw_ptr(), shape,
                                             py_buffer_info.itemsize,
                                             IREE_ALLOCATOR_DEFAULT, &bv),
                 "Error allocating buffer view");

  return HalBufferView::CreateRetained(bv);
}

void SetupRtBindings(pybind11::module m) {
  // BufferPlacement.
  py::enum_<BufferPlacement>(m, "BufferPlacement")
      .value("HEAP", BufferPlacement::kHeap)
      .value("DEVICE_VISIBLE", BufferPlacement::kDeviceVisible)
      .value("DEVICE_LOCAL", BufferPlacement::kDeviceLocal)
      .export_values();

  // RtModule.
  py::class_<RtModule>(m, "Module")
      .def_property_readonly("name", &RtModule::name)
      .def("lookup_function_by_ordinal", &RtModule::lookup_function_by_ordinal)
      .def("lookup_function_by_name", &RtModule::lookup_function_by_name);
  // RtFunction.
  py::class_<RtFunction>(m, "Function")
      .def_property_readonly("name", &RtFunction::name)
      .def_property_readonly("signature", &RtFunction::signature);
  py::class_<iree_rt_function_signature_t>(m, "FunctionSignature")
      .def_readonly("argument_count",
                    &iree_rt_function_signature_t::argument_count)
      .def_readonly("result_count",
                    &iree_rt_function_signature_t::result_count);

  // RtPolicy.
  py::class_<RtPolicy>(m, "Policy").def(py::init(&RtPolicy::Create));

  // RtInstance.
  py::class_<RtInstance>(m, "Instance")
      .def(py::init(&RtInstance::Create),
           py::arg_v("driver_name", absl::optional<std::string>()));

  // RtContext.
  py::class_<RtContext>(m, "Context")
      .def(py::init(&RtContext::Create), py::arg("instance"), py::arg("policy"))
      .def_property_readonly("context_id", &RtContext::context_id)
      .def("register_modules", &RtContext::RegisterModules, py::arg("modules"))
      .def("register_module", &RtContext::RegisterModule, py::arg("module"))
      .def("lookup_module_by_name", &RtContext::LookupModuleByName,
           py::arg("name"))
      .def("resolve_function", &RtContext::ResolveFunction,
           py::arg("full_name"))
      .def("allocate", &RtContext::Allocate, py::arg("allocation_size"),
           py::arg("placement") = BufferPlacement::kHeap,
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("allocate_device_local", &RtContext::AllocateDeviceLocal,
           py::arg("allocation_size"),
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("allocate_device_visible", &RtContext::AllocateDeviceVisible,
           py::arg("allocation_size"),
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("wrap_for_input", &RtContext::WrapPyBufferForInput, py::arg("v"))
      .def("invoke", &RtContext::Invoke, py::arg("f"), py::arg("policy"),
           py::arg("arguments"),
           py::arg("results") = absl::optional<std::vector<HalBufferView*>>());

  // RtInvocation.
  py::class_<RtInvocation>(m, "Invocation")
      .def("query_status", &RtInvocation::QueryStatus)
      .def("await", &RtInvocation::Await,
           py::arg("deadline") = IREE_TIME_INFINITE_FUTURE)
      .def("await_optional", &RtInvocation::AwaitOptional,
           py::arg("deadline") = IREE_TIME_INFINITE_FUTURE)
      .def_property_readonly("results", &RtInvocation::ConsumeResults);
}

}  // namespace python
}  // namespace iree
