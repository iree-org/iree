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
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/path.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"
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

FileHandle FileHandleWrapMemory(py::object host_buffer, bool readable,
                                bool writable, size_t &out_buffer_size) {
  struct Retained {
    Retained(py::object host_buffer)
        : buffer_request(host_buffer, PyBUF_SIMPLE),
          host_buffer(std::move(host_buffer)) {}
    PyBufferRequest buffer_request;
    py::object host_buffer;
  };
  std::unique_ptr<Retained> outer_retained =
      std::make_unique<Retained>(std::move(host_buffer));
  iree_io_file_access_t access = 0;
  if (readable) access |= IREE_IO_FILE_ACCESS_READ;
  if (writable) access |= IREE_IO_FILE_ACCESS_WRITE;
  iree_io_file_handle_t *created_handle;
  out_buffer_size = outer_retained->buffer_request.view().len;
  CheckApiStatus(
      iree_io_file_handle_wrap_host_allocation(
          access,
          iree_byte_span_t{
              static_cast<uint8_t *>(outer_retained->buffer_request.view().buf),
              static_cast<iree_host_size_t>(
                  outer_retained->buffer_request.view().len)},
          iree_io_file_handle_release_callback_t{
              +[](void *user_data, iree_io_file_handle_primitive_t primitive) {
                Retained *inner_retained = static_cast<Retained *>(user_data);
                delete inner_retained;
              },
              (void *)outer_retained.get(),
          },
          iree_allocator_system(), &created_handle),
      "Could not wrap host memory into a file handle");
  outer_retained.release();
  return FileHandle::StealFromRawPtr(created_handle);
}

void ParameterIndexAddFromFileHandle(ParameterIndex &self, std::string &key,
                                     FileHandle &file_handle, uint64_t length,
                                     uint64_t offset,
                                     std::optional<std::string> metadata) {
  iree_io_parameter_index_entry_t entry;
  memset(&entry, 0, sizeof(entry));
  entry.key = iree_make_string_view(key.data(), key.size());
  if (metadata) {
    entry.metadata.data = reinterpret_cast<const uint8_t *>(metadata->data());
    entry.metadata.data_length = metadata->size();
  }
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  entry.storage.file.handle = file_handle.raw_ptr();
  entry.storage.file.offset = offset;
  CheckApiStatus(iree_io_parameter_index_add(self.raw_ptr(), &entry),
                 "Could not add parameter index entry");
}

void ParameterIndexParseFileHandle(ParameterIndex &self,
                                   FileHandle &file_handle,
                                   std::string &format) {
  if (format == "gguf") {
    CheckApiStatus(
        iree_io_parse_gguf_index(file_handle.raw_ptr(), self.raw_ptr()),
        "Could not parse gguf file into index");
  } else if (format == "irpa") {
    CheckApiStatus(
        iree_io_parse_irpa_index(file_handle.raw_ptr(), self.raw_ptr()),
        "Could not parse IREE parameter archive file into index");
  } else if (format == "safetensors") {
    CheckApiStatus(
        iree_io_parse_safetensors_index(file_handle.raw_ptr(), self.raw_ptr()),
        "Could not parse safetensors file into index");
  } else {
    throw std::invalid_argument(
        "Unrecognized file format. Expected one of: 'gguf', 'irpa', "
        "'safetensors'");
  }
}

void ParameterIndexLoadFile(ParameterIndex &self, std::string &file_path,
                            std::optional<std::string> format, bool readable,
                            bool writable, bool mmap) {
  // Default format from extension.
  if (!format) {
    iree_string_view_t path_ext = iree_file_path_extension(
        iree_make_string_view(file_path.data(), file_path.size()));
    format.emplace(path_ext.data, path_ext.size);
  }

  // Open file.
  iree_file_read_flags_t read_flags = IREE_FILE_READ_FLAG_DEFAULT;
  if (mmap) {
    read_flags = IREE_FILE_READ_FLAG_MMAP;
  } else {
    read_flags = IREE_FILE_READ_FLAG_PRELOAD;
  }
  iree_file_contents_t *file_contents = nullptr;
  CheckApiStatus(
      iree_file_read_contents(file_path.c_str(), read_flags,
                              iree_allocator_system(), &file_contents),
      "Error opening parameter file");
  iree_io_file_handle_release_callback_t release_callback = {
      +[](void *user_data, iree_io_file_handle_primitive_t handle_primitive) {
        iree_file_contents_t *file_contents = (iree_file_contents_t *)user_data;
        iree_file_contents_free(file_contents);
      },
      file_contents,
  };

  // Wrap contents.
  iree_io_file_handle_t *raw_file_handle = nullptr;
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, file_contents->buffer, release_callback,
      iree_allocator_system(), &raw_file_handle);
  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(file_contents);
    CheckApiStatus(status, "Error accessing parameter memory");
  }

  // Parse.
  FileHandle file_handle = FileHandle::StealFromRawPtr(raw_file_handle);
  ParameterIndexParseFileHandle(self, file_handle, *format);
}

}  // namespace

void SetupIoBindings(py::module_ &m) {
  m.def("create_io_parameters_module", &CreateIoParametersModule);

  py::class_<FileHandle>(m, "FileHandle")
      .def_static(
          "wrap_memory",
          [](py::object host_buffer, bool readable, bool writable) {
            size_t unused_len;
            return FileHandleWrapMemory(std::move(host_buffer), readable,
                                        writable, unused_len);
          },
          py::arg("host_buffer"), py::arg("readable") = true,
          py::arg("writable") = false);
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
            std::memcpy(entry.storage.splat.pattern, pattern_info.view().buf,
                        pattern_size);
            CheckApiStatus(iree_io_parameter_index_add(self.raw_ptr(), &entry),
                           "Could not add parameter index entry");
          },
          py::arg("key"), py::arg("pattern"), py::arg("total_length"),
          py::arg("metadata") = py::none())
      .def("add_from_file_handle", ParameterIndexAddFromFileHandle,
           py::arg("key"), py::arg("file_handle"), py::arg("length"),
           py::arg("offset") = 0, py::arg("metadata") = py::none())
      .def(
          "add_buffer",
          [](ParameterIndex &self, std::string key, py::object buffer,
             bool readable, bool writable,
             std::optional<std::string> metadata) {
            size_t buffer_size;
            FileHandle file_handle = FileHandleWrapMemory(
                std::move(buffer), readable, writable, buffer_size);
            ParameterIndexAddFromFileHandle(self, key, file_handle, buffer_size,
                                            /*offset=*/0, std::move(metadata));
          },
          py::arg("key"), py::arg("buffer"), py::arg("readable") = true,
          py::arg("writable") = false, py::arg("metadata") = py::none())
      .def("load_from_file_handle", ParameterIndexParseFileHandle,
           py::arg("file_handle"), py::arg("format"))
      .def("load", ParameterIndexLoadFile, py::arg("file_path"),
           py::arg("format") = py::none(), py::arg("readable") = true,
           py::arg("writable") = false, py::arg("mmap") = true)
      .def(
          "create_provider",
          [](ParameterIndex &self, std::string scope,
             std::optional<iree_host_size_t> max_concurrent_operations) {
            if (!max_concurrent_operations) {
              max_concurrent_operations =
                  IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS;
            }
            iree_io_parameter_provider_t *created;
            CheckApiStatus(
                iree_io_parameter_index_provider_create(
                    iree_make_string_view(scope.data(), scope.size()),
                    self.raw_ptr(), *max_concurrent_operations,
                    iree_allocator_system(), &created),
                "Could not create parameter provider from index");
            return ParameterProvider::StealFromRawPtr(created);
          },
          py::arg("scope") = std::string(),
          py::arg("max_concurrent_operations") = py::none());
}

}  // namespace iree::python
