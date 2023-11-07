// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./io.h"

#include <string_view>
#include <unordered_map>

#include "./vm.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"

namespace iree::python {

namespace {

VmModule CreateIoParametersModule(VmInstance &instance, py::args providers) {
  iree_vm_module_t *module = nullptr;
  std::vector<iree_io_parameter_provider_t *> c_providers;
  iree_host_size_t size = providers.size();
  c_providers.resize(size);
  for (iree_host_size_t i = 0; i < size; ++i) {
    ParameterProvider *provider = py::cast<ParameterProvider *>(providers[i]);
    c_providers[i] = provider->raw_ptr();
  }
  CheckApiStatus(iree_io_parameters_module_create(
                     instance.raw_ptr(), size, c_providers.data(),
                     iree_allocator_system(), &module),
                 "Error creating io_parameters module");
  return VmModule::StealFromRawPtr(module);
}

}  // namespace

void SetupIoBindings(py::module_ &m) {
  m.def("create_io_parameters_module", &CreateIoParametersModule);

  py::class_<FileHandle>(m, "FileHandle");
  py::class_<ParameterProvider>(m, "ParameterProvider");
  py::class_<ParameterIndex>(m, "ParameterIndex")
      .def("__init__",
           [](ParameterIndex *new_self) {
             iree_io_parameter_index_t *created;
             CheckApiStatus(iree_io_parameter_index_create(
                                iree_allocator_system(), &created),
                            "Could not create IO parameter index");
             new (new_self) ParameterIndex();
             *new_self = ParameterIndex::StealFromRawPtr(created);
           })
      .def("__len__",
           [](ParameterIndex &self) {
             return iree_io_parameter_index_count(self.raw_ptr());
           })
      .def(
          "reserve",
          [](ParameterIndex &self, iree_host_size_t new_capacity) {
            CheckApiStatus(
                iree_io_parameter_index_reserve(self.raw_ptr(), new_capacity),
                "Could not reserve capacity");
          },
          py::arg("new_capacity"))
      .def(
          "create_provider",
          [](ParameterIndex &self, std::string scope,
             iree_host_size_t max_concurrent_operations) {
            iree_io_parameter_provider_t *created;
            CheckApiStatus(
                iree_io_parameter_index_provider_create(
                    iree_make_string_view(scope.data(), scope.size()),
                    self.raw_ptr(), max_concurrent_operations,
                    iree_allocator_system(), &created),
                "Could not create parameter provider from index");
            return ParameterProvider::StealFromRawPtr(created);
          },
          py::arg("scope") = std::string(),
          py::arg("max_concurrent_operations") =
              IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS);
  //   py::class_<PyModuleInterface>(m, "PyModuleInterface")
  //       .def(py::init<std::string, py::object>(), py::arg("module_name"),
  //            py::arg("ctor"))
  //       .def("__str__", &PyModuleInterface::ToString)
  //       .def_prop_ro("initialized", &PyModuleInterface::initialized)
  //       .def_prop_ro("destroyed", &PyModuleInterface::destroyed)
  //       .def("create", &PyModuleInterface::Create)
  //       .def("export", &PyModuleInterface::ExportFunction, py::arg("name"),
  //            py::arg("cconv"), py::arg("callable"));
}

}  // namespace iree::python
