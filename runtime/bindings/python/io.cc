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
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/schemas/parameter_archive.h"

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

FileHandle FileHandleWrapFd(int fd, bool readable, bool writable) {
  iree_io_file_mode_t mode = 0;
  if (readable) mode |= IREE_IO_FILE_MODE_READ;
  if (writable) mode |= IREE_IO_FILE_MODE_WRITE;
  iree_io_file_handle_t *created_handle;
  CheckApiStatus(iree_io_file_handle_open_fd(mode, fd, iree_allocator_system(),
                                             &created_handle),
                 "Could not wrap host fd into a file handle");

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
  CheckApiStatus(
      iree_io_parse_file_index(
          iree_make_string_view(format.data(), format.size()),
          file_handle.raw_ptr(), self.raw_ptr(), iree_allocator_system()),
      "Could not parse parameter file index");
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

// Wraps an index and an entry, extending lifetime of the index.
struct ParameterIndexEntryWrapper {
  ParameterIndexEntryWrapper(ParameterIndex index) : index(std::move(index)) {}

  ParameterIndex index;
  const iree_io_parameter_index_entry_t *entry = nullptr;
};

}  // namespace

int FileHandle::HandleBufferProtocol(Py_buffer *view, int flags) {
  auto primitive = iree_io_file_handle_primitive(raw_ptr());
  if (primitive.type != IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    PyErr_SetString(PyExc_ValueError,
                    "FileHandle is not based on a host allocation and "
                    "cannot be mapped");
    return -1;
  }
  if (view == NULL) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  view->buf = primitive.value.host_allocation.data;
  view->len = primitive.value.host_allocation.data_length;
  bool is_writable =
      iree_io_file_handle_access(raw_ptr()) & IREE_IO_FILE_ACCESS_WRITE;
  view->readonly = !is_writable;
  view->itemsize = 1;
  view->format = (char *)"B";  // Byte
  view->ndim = 1;
  view->shape = nullptr;
  view->strides = nullptr;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  return 0;
}

void SetupIoBindings(py::module_ &m) {
  m.def("create_io_parameters_module", &CreateIoParametersModule);

  auto file_handle = py::class_<FileHandle>(m, "FileHandle");
  BindBufferProtocol<FileHandle>(file_handle);
  file_handle
      .def_static(
          "wrap_memory",
          [](py::object host_buffer, bool readable, bool writable) {
            size_t unused_len;
            return FileHandleWrapMemory(std::move(host_buffer), readable,
                                        writable, unused_len);
          },
          py::arg("host_buffer"), py::arg("readable") = true,
          py::arg("writable") = false)
      .def_static("wrap_fd", &FileHandleWrapFd, py::arg("fd"),
                  py::arg("readable") = true, py::arg("writable") = false)
      .def_prop_ro(
          "is_host_allocation",
          [](FileHandle &self) {
            auto primitive = iree_io_file_handle_primitive(self.raw_ptr());
            return primitive.type == IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION;
          })
      .def_prop_ro(
          "host_allocation",
          [](py::handle self) {
            return py::steal<py::object>(PyMemoryView_FromObject(self.ptr()));
          })
      .def_prop_ro("is_fd",
                   [](FileHandle &self) {
                     auto primitive =
                         iree_io_file_handle_primitive(self.raw_ptr());
                     return primitive.type == IREE_IO_FILE_HANDLE_TYPE_FD;
                   })
      .def_prop_ro("fd",
                   [](FileHandle &self) {
                     auto primitive =
                         iree_io_file_handle_primitive(self.raw_ptr());
                     return primitive.value.fd;
                   })
      .def("__repr__", [](py::handle self_object) {
        if (py::cast<py::bool_>(self_object.attr("is_host_allocation"))) {
          return py::str("FileHandle<host_allocation({})>")
              .format(self_object.attr("host_allocation"));
        } else if (py::cast<py::bool_>(self_object.attr("is_fd"))) {
          return py::str("FileHandle<fd({})>").format(self_object.attr("fd"));
        } else {
          return py::str("<FileHandle unknown>");
        }
      });

  py::class_<ParameterProvider>(m, "ParameterProvider");
  py::class_<ParameterIndexEntryWrapper>(m, "ParameterIndexEntry")
      .def_prop_ro("key",
                   [](ParameterIndexEntryWrapper &self) {
                     return py::str(self.entry->key.data, self.entry->key.size);
                   })
      .def_prop_ro(
          "length",
          [](ParameterIndexEntryWrapper &self) { return self.entry->length; })
      .def_prop_ro("metadata",
                   [](ParameterIndexEntryWrapper &self) {
                     return py::bytes((const char *)self.entry->metadata.data,
                                      self.entry->metadata.data_length);
                   })
      .def_prop_ro("is_file",
                   [](ParameterIndexEntryWrapper &self) {
                     return self.entry->type ==
                            IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
                   })
      .def_prop_ro("is_splat",
                   [](ParameterIndexEntryWrapper &self) {
                     return self.entry->type ==
                            IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
                   })
      .def_prop_ro(
          "file_storage",
          [](ParameterIndexEntryWrapper &self) {
            if (self.entry->type !=
                IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
              throw std::invalid_argument("Entry is not file storage based");
            }
            return py::make_tuple(
                FileHandle::BorrowFromRawPtr(self.entry->storage.file.handle),
                self.entry->storage.file.offset);
          })
      .def_prop_ro("file_view",
                   [](py::handle self_object) {
                     auto file_storage = self_object.attr("file_storage");
                     py::handle file_handle = file_storage[0];
                     auto offset = py::cast<iree_host_size_t>(file_storage[1]);
                     auto length =
                         py::cast<iree_host_size_t>(self_object.attr("length"));
                     py::object memview = file_handle.attr("host_allocation");
                     py::slice slice(offset, offset + length);
                     return memview.attr("__getitem__")(slice);
                   })
      .def_prop_ro("splat_pattern",
                   [](ParameterIndexEntryWrapper &self) {
                     if (self.entry->type !=
                         IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT) {
                       throw std::invalid_argument("Entry is not splat");
                     }
                     return py::bytes(
                         (const char *)self.entry->storage.splat.pattern,
                         self.entry->storage.splat.pattern_length);
                   })
      .def("__repr__", [](py::handle &self_object) {
        if (py::cast<py::bool_>(self_object.attr("is_splat"))) {
          return py::str("<ParameterIndexEntry '{}' splat {}:{}>")
              .format(self_object.attr("key"),
                      self_object.attr("splat_pattern"),
                      self_object.attr("length"));
        } else if (py::cast<py::bool_>(self_object.attr("is_file"))) {
          py::object file_storage = self_object.attr("file_storage");
          return py::str("<ParameterIndexEntry '{}' {}:{}:{}")
              .format(self_object.attr("key"), file_storage[0], file_storage[1],
                      self_object.attr("length"));
        } else {
          return py::str("<ParameterIndexEntry unknown>");
        }
      });
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
          "__getitem__",
          [](ParameterIndex &self, iree_host_size_t i) {
            ParameterIndexEntryWrapper entry_wrapper(self);
            CheckApiStatus(iree_io_parameter_index_get(self.raw_ptr(), i,
                                                       &entry_wrapper.entry),
                           "Could not enumerate parameter index");
            return entry_wrapper;
          },
          py::arg("i"))
      .def("items",
           [](ParameterIndex &self) {
             py::list items;
             for (iree_host_size_t i = 0;
                  i < iree_io_parameter_index_count(self.raw_ptr()); ++i) {
               ParameterIndexEntryWrapper entry_wrapper(self);
               CheckApiStatus(iree_io_parameter_index_get(self.raw_ptr(), i,
                                                          &entry_wrapper.entry),
                              "Could not enumerate parameter index");
               py::str key(entry_wrapper.entry->key.data,
                           entry_wrapper.entry->key.size);
               py::object value = py::cast(std::move(entry_wrapper));
               items.append(py::make_tuple(key, value));
             }
             return items;
           })
      .def("__repr__",
           [](ParameterIndex &self) {
             iree_string_builder_t b;
             iree_string_builder_initialize(iree_allocator_system(), &b);
             iree_status_t status = iree_io_parameter_index_dump(
                 iree_string_view_empty(), self.raw_ptr(), &b);
             iree_string_view_t sv = iree_string_builder_view(&b);
             py::str result = py::str(sv.data, sv.size);
             iree_string_builder_deinitialize(&b);
             CheckApiStatus(status, "Failed to dump parameter index");
             return result;
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
          py::arg("max_concurrent_operations") = py::none())
      .def(
          "create_archive_file",
          [](ParameterIndex &self, std::string file_path,
             iree_io_physical_offset_t file_offset,
             ParameterIndex *explicit_target_index) {
            // If no target index was given, RAII manage a local target index.
            iree_io_parameter_index_t *target_index = nullptr;
            ParameterIndex default_target_index;
            if (explicit_target_index) {
              target_index = explicit_target_index->raw_ptr();
            } else {
              iree_io_parameter_index_t *created;
              CheckApiStatus(iree_io_parameter_index_create(
                                 iree_allocator_system(), &created),
                             "Could not create IO parameter index");
              default_target_index = ParameterIndex::StealFromRawPtr(created);
              target_index = default_target_index.raw_ptr();
            }

            // Open the file via callback.
            struct OpenParams {
              const char *path;
            };
            OpenParams file_open_user_data{file_path.c_str()};
            auto file_open_callback =
                +[](void *user_data, iree_io_physical_offset_t archive_offset,
                    iree_io_physical_size_t archive_length,
                    iree_io_file_handle_t **out_file_handle) -> iree_status_t {
              OpenParams *params = static_cast<OpenParams *>(user_data);
              iree_file_contents_t *file_contents = NULL;
              IREE_RETURN_IF_ERROR(iree_file_create_mapped(
                  params->path, archive_offset + archive_length, archive_offset,
                  (iree_host_size_t)archive_length, iree_allocator_system(),
                  &file_contents));
              iree_io_file_handle_release_callback_t release_callback;
              memset(&release_callback, 0, sizeof(release_callback));
              release_callback.fn =
                  +[](void *user_data,
                      iree_io_file_handle_primitive_t handle_primitive) {
                    iree_file_contents_free(
                        static_cast<iree_file_contents_t *>(user_data));
                  };
              release_callback.user_data = file_contents;
              iree_status_t status = iree_io_file_handle_wrap_host_allocation(
                  IREE_IO_FILE_ACCESS_WRITE, file_contents->buffer,
                  release_callback, iree_allocator_system(), out_file_handle);
              if (!iree_status_is_ok(status)) {
                iree_file_contents_free(file_contents);
              }
              return status;
            };

            // Write the archive.
            CheckApiStatus(iree_io_build_parameter_archive(
                               self.raw_ptr(), target_index,
                               iree_io_parameter_archive_file_open_callback_t{
                                   file_open_callback,
                                   &file_open_user_data,
                               },
                               file_offset, iree_allocator_system()),
                           "Error building parameter archive");

            // Return the target index.
            return ParameterIndex::BorrowFromRawPtr(target_index);
          },
          py::arg("file_path"), py::arg("file_offset") = 0,
          py::arg("target_index") = nullptr);
}

}  // namespace iree::python
