// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./io.h"

#include <iostream>
#include <string_view>
#include <unordered_map>

#include "./buffer_interop.h"
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
          "add_splat",
          [](ParameterIndex &self, std::string key, py::object pattern,
             uint64_t total_length, std::optional<std::string> metadata) {
            iree_io_parameter_index_entry_t entry;
            memset(&entry, 0, sizeof(entry));
            entry.key = iree_make_string_view(key.data(), key.size());
            if (metadata) {
              entry.metadata.data =
                  reinterpret_cast<const uint8_t *>(metadata->data());
              entry.metadata.data_length = metadata->size();
            }
            entry.length = total_length;
            entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
            PyBufferRequest pattern_info(pattern, PyBUF_SIMPLE);
            auto pattern_size = pattern_info.view().len;
            if (pattern_size > sizeof(entry.storage.splat.pattern)) {
              throw std::invalid_argument(
                  "pattern must be limited to 16 bytes");
            }
            entry.storage.splat.pattern_length = pattern_size;
            // TODO: Do not submit.
            std::memcpy(entry.storage.splat.pattern + 12,
                        pattern_info.view().buf, pattern_size);
            std::memcpy(entry.storage.splat.pattern + 8,
                        pattern_info.view().buf, pattern_size);
            std::memcpy(entry.storage.splat.pattern + 4,
                        pattern_info.view().buf, pattern_size);
            std::memcpy(entry.storage.splat.pattern + 0,
                        pattern_info.view().buf, pattern_size);
            std::cerr << "src[0] = " << std::hex
                      << (uint32_t)((const uint8_t *)pattern_info.view().buf)[0]
                      << std::endl;
            std::cerr << "src[1] = " << std::hex
                      << (uint32_t)((const uint8_t *)pattern_info.view().buf)[1]
                      << std::endl;
            std::cerr << "src[2] = " << std::hex
                      << (uint32_t)((const uint8_t *)pattern_info.view().buf)[2]
                      << std::endl;
            std::cerr << "src[3] = " << std::hex
                      << (uint32_t)((const uint8_t *)pattern_info.view().buf)[3]
                      << std::endl;
            std::cerr << "dst[0] = " << std::hex
                      << (uint32_t)entry.storage.splat.pattern[0] << std::endl;
            std::cerr << "dst[1] = " << std::hex
                      << (uint32_t)entry.storage.splat.pattern[1] << std::endl;
            std::cerr << "dst[2] = " << std::hex
                      << (uint32_t)entry.storage.splat.pattern[2] << std::endl;
            std::cerr << "dst[3] = " << std::hex
                      << (uint32_t)entry.storage.splat.pattern[3] << std::endl;
            CheckApiStatus(iree_io_parameter_index_add(self.raw_ptr(), &entry),
                           "Could not add parameter index entry");
          },
          py::arg("key"), py::arg("pattern"), py::arg("total_length"),
          py::arg("metadata") = py::none())
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
}

}  // namespace iree::python
