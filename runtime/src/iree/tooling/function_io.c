// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/function_io.h"

#include "iree/io/stdio_stream.h"
#include "iree/io/stream.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/numpy_io.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// NOTE: this will get moved at some point but is staged here while it's figured
// out. I'm still not sure how best to factor things so that this doesn't get
// pulled in all the time even if IO is never used. We may end up needing some
// kind of registry that the main iree_io_stream_open uses or allow
// iree_io_file_handle_t to carry a factory function for opening the handles of
// certain types. For now we shim things here at the leaf.
static iree_status_t iree_io_stream_open_path(iree_io_stdio_stream_mode_t mode,
                                              iree_string_view_t path,
                                              uint64_t file_offset,
                                              iree_allocator_t host_allocator,
                                              iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_status_t status = iree_ok_status();
  iree_io_stream_t* stream = NULL;

  status = iree_io_stdio_stream_open(mode, path, host_allocator, &stream);
  if (iree_status_is_ok(status) && file_offset > 0) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, file_offset);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_io_stream_list_t
//===----------------------------------------------------------------------===//

typedef struct iree_io_stream_list_entry_t {
  iree_string_view_t path;
  iree_io_stream_t* stream;
  // + path char storage of path.size
} iree_io_stream_list_entry_t;

// A list of streams indexed by their verbatim paths.
// Streams track their positions and repeated accesses will continue where
// they left off when reading and writing.
//
// NOTE: this is not thread-safe or safe for concurrent access to the same
// streams via different accesses. Re-opening a stream will reset the stream
// offset to 0 and any extant references may end up doing the wrong thing.
typedef struct iree_io_stream_list_t {
  iree_allocator_t host_allocator;
  iree_io_stdio_stream_mode_t mode;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_io_stream_list_entry_t** entries;
} iree_io_stream_list_t;

// Allocates a new stream list where all streams will share the same |mode|.
iree_status_t iree_io_stream_list_allocate(iree_io_stdio_stream_mode_t mode,
                                           iree_allocator_t host_allocator,
                                           iree_io_stream_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_list_t* list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*list), (void**)&list));
  list->host_allocator = host_allocator;
  list->mode = mode;

  *out_list = list;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Frees a stream list and releases all stream resources.
void iree_io_stream_list_free(iree_io_stream_list_t* list) {
  if (!list) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_io_stream_list_entry_t* entry = list->entries[i];
    iree_io_stream_release(entry->stream);
    iree_allocator_free(list->host_allocator, entry);
  }
  iree_allocator_free(list->host_allocator, list->entries);

  iree_allocator_free(list->host_allocator, list);

  IREE_TRACE_ZONE_END(z0);
}

// Returns the entry matching the given verbatim |path| or NULL if not found.
static iree_io_stream_list_entry_t* iree_io_stream_list_find_entry(
    iree_io_stream_list_t* list, iree_string_view_t path) {
  IREE_ASSERT_ARGUMENT(list);
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_io_stream_list_entry_t* entry = list->entries[i];
    if (iree_string_view_equal(path, entry->path)) {
      return entry;
    }
  }
  return NULL;
}

// Appends a stream to the list. The |path| will be cloned into the list
// storage and the stream will be retained until the list is freed.
static iree_status_t iree_io_stream_list_append_entry(
    iree_io_stream_list_t* list, iree_string_view_t path,
    iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // Grow if needed.
  if (list->count + 1 > list->capacity) {
    iree_host_size_t new_capacity = iree_max(list->capacity * 2, 16);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(list->host_allocator,
                                   new_capacity * sizeof(*list->entries[0]),
                                   (void**)&list->entries));
    list->capacity = new_capacity;
  }

  // Allocate the entry and its string storage.
  iree_io_stream_list_entry_t* entry = NULL;
  iree_status_t status = iree_allocator_malloc(
      list->host_allocator, sizeof(*entry) + path.size, (void**)&entry);
  if (iree_status_is_ok(status)) {
    entry->path.data = (const char*)entry + sizeof(*entry);
    entry->path.size = path.size;
    memcpy((void*)entry->path.data, path.data, path.size);
    entry->stream = stream;
    iree_io_stream_retain(entry->stream);
  }

  // Store in list.
  if (iree_status_is_ok(status)) {
    list->entries[list->count++] = entry;
  } else {
    iree_allocator_free(list->host_allocator, entry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Opens an |entry| that has already been opened, resetting its position to
// 0 if needed or directly returning it if appending.
static iree_status_t iree_io_stream_list_open_existing(
    iree_io_stream_list_t* list, iree_string_view_t path,
    iree_io_stream_list_entry_t* entry, bool is_append,
    iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_status_t status = iree_ok_status();

  // Reset the stream back to 0 if not appending. Today we require seekable
  // streams for this. We could re-open the file but the only non-seekable
  // stream we have today is stdin and reopening that isn't possible without
  // buffering that we don't/won't do.
  if (!is_append) {
    if (iree_all_bits_set(iree_io_stream_mode(entry->stream),
                          IREE_IO_STREAM_MODE_SEEKABLE)) {
      status = iree_io_stream_seek(entry->stream, IREE_IO_STREAM_SEEK_SET, 0);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "opened stream from `%.*s` is not seekable and cannot be reopened",
          (int)path.size, path.data);
    }
  }

  iree_io_stream_retain(entry->stream);
  *out_stream = entry->stream;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Opens a stream at |path|, possibly reusing an existing stream if it has
// already been opened. If |append| is true the stream will be returned at its
// existing position for continued reading/writing and otherwise it will be
// reset to position 0.
iree_status_t iree_io_stream_list_open(iree_io_stream_list_t* list,
                                       iree_string_view_t path, bool is_append,
                                       iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // Lookup an existing entry - if found we can reuse it.
  iree_io_stream_list_entry_t* entry =
      iree_io_stream_list_find_entry(list, path);
  if (entry) {
    iree_status_t status = iree_io_stream_list_open_existing(
        list, path, entry, is_append, out_stream);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Open the file at the path specified.
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_open_path(list->mode, path, 0ull, list->host_allocator,
                                   &stream));

  // Append the stream entry to the list so it's retained for future opens.
  iree_status_t status = iree_io_stream_list_append_entry(list, path, stream);

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_consume_cconv_any(iree_string_view_t* cconv,
                                                    char* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = 0;
  if (cconv->size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function expected fewer input values");
  }
  *out_type = cconv->data[0];
  ++cconv->data;
  --cconv->size;
  return iree_ok_status();
}

static iree_status_t iree_tooling_consume_cconv(iree_string_view_t* cconv,
                                                char expected_type) {
  char actual_type = 0;
  IREE_RETURN_IF_ERROR(iree_tooling_consume_cconv_any(cconv, &actual_type));
  if (actual_type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function signature mismatch: expected cconv type "
                            "`%c` but provided type `%c`",
                            expected_type, actual_type);
  }
  return iree_ok_status();
}

static iree_status_t iree_tooling_parse_null_into(iree_string_view_t* cconv,
                                                  iree_string_view_t string,
                                                  iree_vm_list_t* list) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, string.data, string.size);

  // Get the expected cconv type so we can handle 0 for primitives and
  // NULL for refs.
  char cconv_type = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_consume_cconv_any(cconv, &cconv_type));

  // Add the appropriate variant type to the list.
  iree_status_t status = iree_ok_status();
  switch (cconv_type) {
    case IREE_VM_CCONV_TYPE_I32: {
      iree_vm_value_t value = iree_vm_value_make_i32(0);
      status = iree_vm_list_push_value(list, &value);
      break;
    }
    case IREE_VM_CCONV_TYPE_F32: {
      iree_vm_value_t value = iree_vm_value_make_f32(0.0f);
      status = iree_vm_list_push_value(list, &value);
      break;
    }
    case IREE_VM_CCONV_TYPE_I64: {
      iree_vm_value_t value = iree_vm_value_make_i64(0ll);
      status = iree_vm_list_push_value(list, &value);
      break;
    }
    case IREE_VM_CCONV_TYPE_F64: {
      iree_vm_value_t value = iree_vm_value_make_f64(0.0);
      status = iree_vm_list_push_value(list, &value);
      break;
    }
    case IREE_VM_CCONV_TYPE_REF: {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      status = iree_vm_list_push_ref_retain(list, &null_ref);
      break;
    }
    default: {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented cconv type `%c`", cconv_type);
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_parse_primitive_into(
    iree_string_view_t* cconv, iree_string_view_t string, iree_vm_list_t* list,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, string.data, string.size);

  // Get the expected cconv type to help us parse the primitive value.
  char cconv_type = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_consume_cconv_any(cconv, &cconv_type));

  iree_status_t status = iree_ok_status();
  switch (cconv_type) {
    case IREE_VM_CCONV_TYPE_I32: {
      iree_vm_value_t value = iree_vm_value_make_i32(0);
      if (iree_string_view_atoi_int32(string, &value.i32)) {
        status = iree_vm_list_push_value(list, &value);
      } else {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value `%.*s` as i32",
                                  (int)string.size, string.data);
      }
      break;
    }
    case IREE_VM_CCONV_TYPE_F32: {
      iree_vm_value_t value = iree_vm_value_make_f32(0.0f);
      if (iree_string_view_atof(string, &value.f32)) {
        status = iree_vm_list_push_value(list, &value);
      } else {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value `%.*s` as f32",
                                  (int)string.size, string.data);
      }
      break;
    }
    case IREE_VM_CCONV_TYPE_I64: {
      iree_vm_value_t value = iree_vm_value_make_i64(0ll);
      if (iree_string_view_atoi_int64(string, &value.i64)) {
        status = iree_vm_list_push_value(list, &value);
      } else {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value `%.*s` as i64",
                                  (int)string.size, string.data);
      }
      break;
    }
    case IREE_VM_CCONV_TYPE_F64: {
      iree_vm_value_t value = iree_vm_value_make_f64(0.0);
      if (iree_string_view_atod(string, &value.f64)) {
        status = iree_vm_list_push_value(list, &value);
      } else {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value `%.*s` as f64",
                                  (int)string.size, string.data);
      }
      break;
    }
    default: {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unimplemented cconv type `%c`", cconv_type);
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_parse_buffer_view_file_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_io_stream_t* stream = (iree_io_stream_t*)user_data;
  return iree_io_stream_read(stream, mapping->contents.data_length,
                             mapping->contents.data,
                             /*out_buffer_length=*/NULL);
}

// Creates a HAL buffer view with the given |metadata| and reads the contents
// from the file reference in |string| which has the prefix `@` to indicate
// the contents starting from 0 and `+` for the next contents in an already
// opened stream.
// The file contents are directly read in to memory with no processing.
static iree_status_t iree_tooling_parse_buffer_view_file(
    iree_string_view_t metadata, iree_string_view_t string,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_io_stream_list_t* stream_list,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, string.data, string.size);

  // Parse shape and element type used to allocate the buffer view.
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t shape[128] = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_parse_shape_and_element_type(
                                            metadata, IREE_ARRAYSIZE(shape),
                                            &shape_rank, shape, &element_type));

  // TODO(benvanik): allow specifying the encoding.
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  // @ = open new
  // + = append
  bool is_append = !iree_string_view_starts_with(string, IREE_SV("@"));
  iree_string_view_t path =
      iree_string_view_substr(string, 1, IREE_HOST_SIZE_MAX);

  // Open (or retrieve) the file.
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_list_open(stream_list, path, is_append, &stream));

  // TODO(benvanik): support mapping on allocators that can handle importing
  // host memory. We only want to do this when it won't hurt performance
  // (unified memory systems and where the mapped memory meets alignment
  // requirements). For now we always wire new device memory to read into.
  // A real application would want to either do the import or use
  // iree_hal_file_t to stream the file contents into device memory without
  // going through host memory.

  // Read the stream contents into the buffer.
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      device, device_allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params, iree_tooling_parse_buffer_view_file_callback, stream,
      out_buffer_view);

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Parses a shaped tensor type into a HAL buffer view.
static iree_status_t iree_tooling_parse_tensor(
    iree_string_view_t string, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_io_stream_list_t* stream_list,
    iree_allocator_t host_allocator, iree_hal_buffer_view_t** out_buffer_view) {
  // If contents are sourced from a file then route to that, and otherwise
  // parse as a normal HAL buffer view with inline contents (or none).
  iree_string_view_t metadata, contents;
  if (iree_string_view_split(string, '=', &metadata, &contents) != -1) {
    if (iree_string_view_starts_with(contents, IREE_SV("@")) ||
        iree_string_view_starts_with(contents, IREE_SV("+"))) {
      return iree_tooling_parse_buffer_view_file(metadata, contents, device,
                                                 device_allocator, stream_list,
                                                 out_buffer_view);
    }
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, string.data, string.size);
  iree_status_t status = iree_hal_buffer_view_parse(
      string, device, device_allocator, out_buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Parses a shaped tensor type into a HAL buffer view and appends it to |list|.
static iree_status_t iree_tooling_parse_tensor_into(
    iree_string_view_t* cconv, iree_string_view_t string, iree_vm_list_t* list,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_io_stream_list_t* stream_list, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Expect a ref holding the buffer view.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_tooling_consume_cconv(cconv, 'r'));

  // Expect tensors to have a shape/type. Kinda sketchy but filters out some
  // typos (scalar values).
  bool has_equal =
      iree_string_view_find_char(string, '=', 0) != IREE_STRING_VIEW_NPOS;
  bool has_x =
      iree_string_view_find_char(string, 'x', 0) != IREE_STRING_VIEW_NPOS;
  if (!has_equal && !has_x) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid tensor specification, requires at least a "
                            "shape/type (`4x2xf32=...`)");
  }

  // Parse the tensor contents.
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_parse_tensor(string, device, device_allocator,
                                    stream_list, host_allocator, &buffer_view));

  // Add buffer view to list.
  iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
  iree_status_t status = iree_vm_list_push_ref_retain(list, &buffer_view_ref);

  iree_hal_buffer_view_release(buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Parses a shaped tensor type with optional contents to size a HAL buffer for
// output storage and appends it to |list|.
static iree_status_t iree_tooling_parse_storage_into(
    iree_string_view_t* cconv, iree_string_view_t string, iree_vm_list_t* list,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_io_stream_list_t* stream_list, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Expect a ref holding the buffer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_tooling_consume_cconv(cconv, 'r'));

  // Parse the tensor contents.
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_parse_tensor(string, device, device_allocator,
                                    stream_list, host_allocator, &buffer_view));

  // Add just the storage buffer to the list - we don't need the metadata.
  iree_vm_ref_t buffer_ref =
      iree_hal_buffer_move_ref(iree_hal_buffer_view_buffer(buffer_view));
  iree_status_t status = iree_vm_list_push_ref_retain(list, &buffer_ref);

  iree_hal_buffer_view_release(buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Parses a single ndarray from |stream| as a HAL buffer view and appends it to
// |list|.
static iree_status_t iree_tooling_parse_ndarray_into(
    iree_string_view_t* cconv, iree_vm_list_t* list, iree_io_stream_t* stream,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Expect a ref holding the buffer view.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_tooling_consume_cconv(cconv, 'r'));

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_READ,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status = iree_numpy_npy_load_ndarray(
      stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params, device,
      device_allocator, &buffer_view);

  if (iree_status_is_ok(status)) {
    iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
    status = iree_vm_list_push_ref_retain(list, &buffer_view_ref);
  }

  iree_hal_buffer_view_release(buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Parses zero or more variants from a file into the |list|.
// The |string| defines the file mode (`@` new, `+` existing, `*` splat) and
// the path to source from.
static iree_status_t iree_tooling_parse_file_into(
    iree_string_view_t* cconv, iree_string_view_t string, iree_vm_list_t* list,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_io_stream_list_t* stream_list, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, string.data, string.size);

  // @ = open new
  // +/* = append
  // * = splat
  bool is_append = !iree_string_view_starts_with(string, IREE_SV("@"));
  bool is_splat = iree_string_view_starts_with(string, IREE_SV("*"));
  iree_string_view_t path =
      iree_string_view_substr(string, 1, IREE_HOST_SIZE_MAX);

  // Today we only support numpy files here but could make this pluggable or at
  // least a little smarter (sniff file header/etc) instead of relying on ext.
  if (!iree_string_view_ends_with(path, IREE_SV(".npy"))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "only numpy (.npy) files are supported for metadata-less variant I/O");
  }

  // Open (or retrieve) the file.
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_list_open(stream_list, path, is_append, &stream));

  iree_status_t status = iree_ok_status();
  if (!is_splat) {
    // Read a single ndarray from the stream at the current offset.
    status = iree_tooling_parse_ndarray_into(cconv, list, stream, device,
                                             device_allocator);
  } else {
    // Read zero or more ndarrays from the stream - note that it may already be
    // at EOS.
    while (iree_status_is_ok(status) && !iree_io_stream_is_eos(stream)) {
      status = iree_tooling_parse_ndarray_into(cconv, list, stream, device,
                                               device_allocator);
    }
  }

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_parse_variant_into(
    iree_string_view_t* cconv, iree_string_view_t string, iree_vm_list_t* list,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_io_stream_list_t* stream_list, iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(string)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no value specified for input");
  } else if (iree_string_view_equal(string, IREE_SV("(null)")) ||
             iree_string_view_equal(string, IREE_SV("(ignored)"))) {
    return iree_tooling_parse_null_into(cconv, string, list);
  } else if (iree_string_view_starts_with(string, IREE_SV("@")) ||
             iree_string_view_starts_with(string, IREE_SV("+")) ||
             iree_string_view_starts_with(string, IREE_SV("*"))) {
    return iree_tooling_parse_file_into(cconv, string, list, device,
                                        device_allocator, stream_list,
                                        host_allocator);
  } else if (iree_string_view_consume_prefix(&string, IREE_SV("&"))) {
    return iree_tooling_parse_storage_into(cconv, string, list, device,
                                           device_allocator, stream_list,
                                           host_allocator);
  } else if (!iree_string_view_starts_with(*cconv, IREE_SV("r"))) {
    return iree_tooling_parse_primitive_into(cconv, string, list, device,
                                             device_allocator, host_allocator);
  }
  // Shaped tensor as a buffer view.
  // NOTE: we could support more things here - strings, VM buffers or lists,
  // etc. Today if it's not a null or primitive value it's a tensor.
  return iree_tooling_parse_tensor_into(cconv, string, list, device,
                                        device_allocator, stream_list,
                                        host_allocator);
}

static iree_status_t iree_tooling_parse_variants_into(
    iree_string_view_t cconv, iree_string_view_list_t specs,
    iree_vm_list_t* list, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_TRACE_ZONE_BEGIN(z0);

  // List of opened streams used for allowing multiple arguments to source from
  // the same file sequentially.
  iree_io_stream_list_t* stream_list = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_list_allocate(
      IREE_IO_STDIO_STREAM_MODE_READ, host_allocator, &stream_list));

  // Parse each variant string. Note that some strings may expand to zero or
  // more variants and so we need to consume the cconv based on how many were
  // parsed.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < specs.count; ++i) {
    iree_string_view_t string = iree_string_view_trim(specs.values[i]);
    status = iree_status_annotate_f(
        iree_tooling_parse_variant_into(&cconv, string, list, device,
                                        device_allocator, stream_list,
                                        host_allocator),
        "parsing input `%.*s`", (int)string.size, string.data);
    if (!iree_status_is_ok(status)) break;
  }

  iree_io_stream_list_free(stream_list);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_parse_variants(
    iree_string_view_t cconv, iree_string_view_list_t specs,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_vm_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Argument list that will be populated - possibly returning with 0 entries.
  iree_vm_list_t* list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_list_create(iree_vm_make_undefined_type_def(), specs.count,
                              host_allocator, &list));

  // Parse into the argument list.
  iree_status_t status = iree_tooling_parse_variants_into(
      cconv, specs, list, device, device_allocator, host_allocator);

  if (iree_status_is_ok(status)) {
    *out_list = list;
  } else {
    iree_vm_list_release(list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

#define IREE_PRINT_VARIANT_CASE_I(SIZE, B, V)                                \
  case IREE_VM_VALUE_TYPE_I##SIZE:                                           \
    status = iree_string_builder_append_format(B, "i" #SIZE "=%" PRIi##SIZE, \
                                               (V).i##SIZE);                 \
    break;

#define IREE_PRINT_VARIANT_CASE_F(SIZE, B, V)                               \
  case IREE_VM_VALUE_TYPE_F##SIZE:                                          \
    status =                                                                \
        iree_string_builder_append_format(B, "f" #SIZE "=%g", (V).f##SIZE); \
    break;

static iree_status_t iree_tooling_format_variant(
    iree_vm_variant_t variant, iree_host_size_t max_element_count,
    iree_string_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (iree_vm_variant_is_empty(variant)) {
    status = iree_string_builder_append_string(builder, IREE_SV("(null)"));
  } else if (iree_vm_variant_is_value(variant)) {
    switch (iree_vm_type_def_as_value(variant.type)) {
      IREE_PRINT_VARIANT_CASE_I(8, builder, variant)
      IREE_PRINT_VARIANT_CASE_I(16, builder, variant)
      IREE_PRINT_VARIANT_CASE_I(32, builder, variant)
      IREE_PRINT_VARIANT_CASE_I(64, builder, variant)
      IREE_PRINT_VARIANT_CASE_F(32, builder, variant)
      IREE_PRINT_VARIANT_CASE_F(64, builder, variant)
      default:
        status = iree_string_builder_append_string(builder, IREE_SV("?"));
        break;
    }
  } else if (iree_vm_variant_is_ref(variant)) {
    iree_string_view_t type_name =
        iree_vm_ref_type_name(iree_vm_type_def_as_ref(variant.type));
    status = iree_string_builder_append_string(builder, type_name);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(builder, IREE_SV("\n"));
    }
    if (iree_status_is_ok(status)) {
      if (iree_vm_list_isa(variant.ref)) {
        iree_vm_list_t* child_list = iree_vm_list_deref(variant.ref);
        status = iree_tooling_format_variants(IREE_SV("child_list"), child_list,
                                              max_element_count, builder);
      } else if (iree_hal_buffer_view_isa(variant.ref)) {
        iree_hal_buffer_view_t* buffer_view =
            iree_hal_buffer_view_deref(variant.ref);
        status = iree_hal_buffer_view_append_to_builder(
            buffer_view, max_element_count, builder);
      } else {
        // TODO(benvanik): a way for ref types to describe themselves.
        status =
            iree_string_builder_append_string(builder, IREE_SV("(no printer)"));
      }
    }
  } else {
    status = iree_string_builder_append_string(builder, IREE_SV("(null)"));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_format_variants(iree_string_view_t list_name,
                                           iree_vm_list_t* list,
                                           iree_host_size_t max_element_count,
                                           iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_variant_assign(list, i, &variant));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_string_builder_append_format(
                builder, "%.*s[%" PRIhsz "]: ", (int)list_name.size,
                list_name.data, i));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tooling_format_variant(variant, max_element_count, builder));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_print_variant(
    iree_vm_variant_t variant, iree_host_size_t max_element_count,
    iree_io_stream_t* stream, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  iree_status_t status =
      iree_tooling_format_variant(variant, max_element_count, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_write(stream, iree_string_builder_size(&builder),
                                  iree_string_builder_buffer(&builder));
  }

  iree_string_builder_deinitialize(&builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_print_variants(iree_string_view_t list_name,
                                          iree_vm_list_t* list,
                                          iree_host_size_t max_element_count,
                                          iree_io_stream_t* stream,
                                          iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reused across each variant print to amortize allocations.
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    status = iree_vm_list_get_variant_assign(list, i, &variant);
    if (!iree_status_is_ok(status)) break;
    status = iree_string_builder_append_format(
        &builder, "%.*s[%" PRIhsz "]: ", (int)list_name.size, list_name.data,
        i);
    if (!iree_status_is_ok(status)) break;
    status = iree_tooling_format_variant(variant, max_element_count, &builder);
    if (!iree_status_is_ok(status)) break;
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
    if (!iree_status_is_ok(status)) break;
    status = iree_io_stream_write(stream, iree_string_builder_size(&builder),
                                  iree_string_builder_buffer(&builder));
    if (!iree_status_is_ok(status)) break;
    iree_string_builder_reset(&builder);
  }

  iree_string_builder_deinitialize(&builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Writing
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_create_buffer_view_with_hal_buffer(
    iree_hal_buffer_t* hal_buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  iree_hal_dim_t shape[1] = {
      (iree_hal_dim_t)iree_hal_buffer_byte_length(hal_buffer),
  };
  return iree_hal_buffer_view_create(
      hal_buffer, IREE_ARRAYSIZE(shape), shape, IREE_HAL_ELEMENT_TYPE_INT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, host_allocator, out_buffer_view);
}

static void iree_hal_buffer_release_vm_buffer(
    void* user_data, struct iree_hal_buffer_t* buffer) {
  iree_vm_buffer_release((iree_vm_buffer_t*)user_data);
}

static iree_status_t iree_tooling_create_buffer_view_with_vm_buffer(
    iree_vm_buffer_t* vm_buffer, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_view_t** out_buffer_view) {
  // Get read-only pointer to the underlying buffer heap memory.
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(
      vm_buffer, 0, iree_vm_buffer_length(vm_buffer), 1, &span));

  // Wrap the heap memory in a HAL buffer for read-only access.
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_buffer_release_vm_buffer,
      .user_data = vm_buffer,
  };
  iree_vm_buffer_retain(vm_buffer);
  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_heap_buffer_wrap(
      device_allocator, IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE | IREE_HAL_BUFFER_USAGE_MAPPING,
      span.data_length, iree_cast_const_byte_span(span), release_callback,
      &hal_buffer);
  iree_vm_buffer_release(vm_buffer);

  // Wrap the HAL buffer in a buffer view.
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_buffer_view_with_hal_buffer(
        hal_buffer, host_allocator, out_buffer_view);
  }

  iree_hal_buffer_release(hal_buffer);
  return status;
}

static iree_status_t iree_tooling_create_buffer_view_empty(
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  iree_hal_buffer_t* hal_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
      device_allocator, IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE | IREE_HAL_BUFFER_USAGE_MAPPING, 0,
      iree_byte_span_empty(), iree_hal_buffer_release_callback_null(),
      &hal_buffer));
  iree_status_t status = iree_tooling_create_buffer_view_with_hal_buffer(
      hal_buffer, host_allocator, out_buffer_view);
  iree_hal_buffer_release(hal_buffer);
  return status;
}

static iree_status_t iree_tooling_create_buffer_view_with_value(
    iree_vm_value_t value, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_view_t** out_buffer_view) {
  iree_device_size_t byte_length = 0;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_NONE:
      return iree_tooling_create_buffer_view_empty(
          device_allocator, host_allocator, out_buffer_view);
    case IREE_VM_VALUE_TYPE_I8:
      byte_length = sizeof(value.i8);
      element_type = IREE_HAL_ELEMENT_TYPE_INT_8;
      break;
    case IREE_VM_VALUE_TYPE_I16:
      byte_length = sizeof(value.i16);
      element_type = IREE_HAL_ELEMENT_TYPE_INT_16;
      break;
    case IREE_VM_VALUE_TYPE_I32:
      byte_length = sizeof(value.i32);
      element_type = IREE_HAL_ELEMENT_TYPE_INT_32;
      break;
    case IREE_VM_VALUE_TYPE_I64:
      byte_length = sizeof(value.i64);
      element_type = IREE_HAL_ELEMENT_TYPE_INT_64;
      break;
    case IREE_VM_VALUE_TYPE_F32:
      byte_length = sizeof(value.f32);
      element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      break;
    case IREE_VM_VALUE_TYPE_F64:
      byte_length = sizeof(value.f64);
      element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported value type");
  }

  iree_hal_buffer_params_t params = {
      .usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
  };
  iree_hal_buffer_t* hal_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, byte_length, &hal_buffer));

  iree_status_t status = iree_hal_buffer_map_write(
      hal_buffer, 0, value.value_storage, byte_length);

  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_create(hal_buffer, /*shape_rank=*/0,
                                         /*shape=*/NULL, element_type,
                                         IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                         host_allocator, out_buffer_view);
  }

  iree_hal_buffer_release(hal_buffer);
  return status;
}

static iree_status_t iree_tooling_create_buffer_view_from_variant(
    iree_vm_variant_t variant, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;
  if (iree_vm_variant_is_empty(variant)) {
    // Empty value - we need to emit a zero-length value to keep the npy file
    // ordered when there are multiple entries.
    return iree_tooling_create_buffer_view_empty(
        device_allocator, host_allocator, out_buffer_view);
  } else if (iree_vm_variant_is_ref(variant)) {
    if (iree_hal_buffer_view_isa(variant.ref)) {
      // Buffer view returned can provide the metadata required.
      *out_buffer_view = iree_hal_buffer_view_deref(variant.ref);
      iree_hal_buffer_view_retain(*out_buffer_view);
      return iree_ok_status();
    } else if (iree_hal_buffer_isa(variant.ref)) {
      // i8 buffer view of the total length of the HAL buffer.
      iree_hal_buffer_t* buffer = iree_hal_buffer_deref(variant.ref);
      return iree_tooling_create_buffer_view_with_hal_buffer(
          buffer, host_allocator, out_buffer_view);
    } else if (iree_vm_buffer_isa(variant.ref)) {
      // i8 buffer view of the total length of the VM buffer wrapped in a HAL
      // buffer.
      iree_vm_buffer_t* buffer = iree_vm_buffer_deref(variant.ref);
      return iree_tooling_create_buffer_view_with_vm_buffer(
          buffer, device_allocator, host_allocator, out_buffer_view);
    } else {
      // Unsupported type.
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported output source type; expected: "
                              "!hal.buffer, !hal.buffer_view, !vm.buffer");
    }
  } else {
    // Primitive value that we wrap in a scalar buffer view.
    return iree_tooling_create_buffer_view_with_value(
        iree_vm_variant_value(variant), device_allocator, host_allocator,
        out_buffer_view);
  }
}

static iree_status_t iree_tooling_write_variant_to_npy_file(
    iree_io_stream_t* stream, iree_vm_variant_t variant,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator) {
  // npy files require buffer views so if we receive anything but a buffer view
  // we wrap it in one typed as bytes.
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_create_buffer_view_from_variant(
      variant, device_allocator, host_allocator, &buffer_view));

  // Append buffer view contents to the file stream.
  iree_numpy_npy_save_options_t options = IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT;
  iree_status_t status =
      iree_numpy_npy_save_ndarray(stream, options, buffer_view, host_allocator);

  iree_hal_buffer_view_release(buffer_view);
  return status;
}

static iree_status_t iree_tooling_write_variant_to_binary_file(
    iree_io_stream_t* stream, iree_vm_variant_t variant,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator) {
  // Today we reuse the buffer view code to get the variant into a byte buffer
  // to write out even though we don't use any of the metadata. This is a
  // command line tool writing out files using stdio and not an example of how
  // to create a high performance I/O mechanism.
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_create_buffer_view_from_variant(
      variant, device_allocator, host_allocator, &buffer_view));
  iree_device_size_t byte_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  // Map the buffer memory into a host pointer so we can access it.
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &mapping);

  // Write to the file from the mapped memory.
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_write(stream, byte_length, mapping.contents.data);
  }

  iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_view_release(buffer_view);
  return status;
}

static iree_status_t iree_tooling_write_variant_to_file(
    iree_vm_variant_t variant, iree_string_view_t spec,
    iree_allocator_t host_allocator) {
  // Open the output file based on the spec.
  iree_io_stdio_stream_mode_t mode = 0;
  if (iree_string_view_consume_prefix(&spec, IREE_SV("@"))) {
    mode |= IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD;
  } else if (iree_string_view_consume_prefix(&spec, IREE_SV("+"))) {
    mode |= IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_APPEND;
  } else {
    // We only support overwrite and append for now.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported output mode specification '%.*s'",
                            (int)spec.size, spec.data);
  }
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_stdio_stream_open(mode, spec, host_allocator, &stream));

  // Dummy heap used for allocating transient variants.
  // This is wasteful to cycle for each one but we don't care about it in the
  // tooling.
  iree_hal_allocator_t* device_allocator = NULL;
  iree_status_t status = iree_hal_allocator_create_heap(
      IREE_SV("tooling"), host_allocator, host_allocator, &device_allocator);

  // Output format is based on file extension with ones we don't know about
  // going into binary mode. Some formats require metadata from buffer views
  // but in binary mode we just dump whatever contents we have and leave it up
  // to the user to handle the shape/type/encoding.
  if (iree_status_is_ok(status)) {
    if (iree_string_view_ends_with(spec, IREE_SV(".npy"))) {
      status = iree_tooling_write_variant_to_npy_file(
          stream, variant, device_allocator, host_allocator);
    } else {
      status = iree_tooling_write_variant_to_binary_file(
          stream, variant, device_allocator, host_allocator);
    }
  }

  iree_hal_allocator_release(device_allocator);
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t iree_tooling_write_variant(
    iree_vm_variant_t variant, iree_string_view_t spec,
    iree_host_size_t max_element_count, iree_io_stream_t* default_stream,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (iree_string_view_is_empty(spec)) {
    // Send into the void.
  } else if (iree_string_view_equal(spec, IREE_SV("-"))) {
    // Route to the default stream (if provided).
    if (default_stream) {
      status = iree_tooling_print_variant(variant, max_element_count,
                                          default_stream, host_allocator);
    }
  } else {
    // Write to a file.
    status = iree_tooling_write_variant_to_file(variant, spec, host_allocator);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_write_variants(iree_vm_list_t* list,
                                          iree_string_view_list_t specs,
                                          iree_host_size_t max_element_count,
                                          iree_io_stream_t* default_stream,
                                          iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_vm_list_size(list) != specs.count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%" PRIhsz
        " outputs specified but the provided variant list only has %" PRIhsz
        " elements",
        specs.count, iree_vm_list_size(list));
  }

  for (iree_host_size_t i = 0; i < specs.count; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_vm_list_get_variant_assign(list, i, &variant));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_tooling_write_variant(variant, specs.values[i], max_element_count,
                                   default_stream, host_allocator));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
