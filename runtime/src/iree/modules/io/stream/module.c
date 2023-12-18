// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/io/stream/module.h"

#define IREE_IO_STREAM_MODULE_VERSION_0_0 0x00000000u
#define IREE_IO_STREAM_MODULE_VERSION_LATEST IREE_IO_STREAM_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// iree_io_file_stream_t
//===----------------------------------------------------------------------===//
// TODO(benvanik): make this use iree_io_file_handle_t and move there.
// This here is just a partial implementation for use with console IO.

#if IREE_FILE_IO_ENABLE

#if defined(IREE_PLATFORM_WINDOWS)
#define iree_fseek64 _fseeki64
#define iree_ftell64 _ftelli64
#else
#define iree_fseek64 fseeko
#define iree_ftell64 ftello
#endif  // IREE_PLATFORM_WINDOWS

typedef struct iree_io_file_stream_t {
  iree_io_stream_t base;
  iree_allocator_t host_allocator;
  iree_io_stream_release_callback_t release_callback;
  FILE* handle;
} iree_io_file_stream_t;

static const iree_io_stream_vtable_t iree_io_file_stream_vtable;

static iree_io_file_stream_t* iree_io_file_stream_cast(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  return (iree_io_file_stream_t*)base_stream;
}

IREE_API_EXPORT iree_status_t iree_io_file_stream_wrap(
    iree_io_stream_mode_t mode, FILE* handle,
    iree_io_stream_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_file_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &iree_io_file_stream_vtable;
  stream->base.mode = mode;
  stream->host_allocator = host_allocator;
  stream->release_callback = release_callback;
  stream->handle = handle;

  *out_stream = &stream->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_io_file_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  iree_allocator_t host_allocator = stream->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream->release_callback.fn) {
    stream->release_callback.fn(stream->release_callback.user_data,
                                base_stream);
  }

  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

static iree_io_stream_pos_t iree_io_file_stream_offset(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  return (iree_io_stream_pos_t)iree_ftell64(stream->handle);
}

static iree_io_stream_pos_t iree_io_file_stream_length(
    iree_io_stream_t* base_stream) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);

  // Capture original offset so we can return to it.
  uint64_t origin = iree_ftell64(stream->handle);

  // Seek to the end of the file.
  if (iree_fseek64(stream->handle, 0, SEEK_END) == -1) {
    return -1;
  }

  // Query the position, telling us the total file length in bytes.
  uint64_t file_length = iree_ftell64(stream->handle);
  if (file_length == -1L) {
    return -1;
  }

  // Seek back to the file origin.
  if (iree_fseek64(stream->handle, origin, SEEK_SET) == -1) {
    return -1;
  }

  return (iree_io_stream_pos_t)file_length;
}

static iree_status_t iree_io_file_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t offset) {
  IREE_ASSERT_ARGUMENT(base_stream);
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  int origin = 0;
  switch (seek_mode) {
    default:
    case IREE_IO_STREAM_SEEK_SET:
      origin = SEEK_SET;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      origin = SEEK_CUR;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      origin = SEEK_END;
      break;
  }

  iree_status_t status = iree_ok_status();
  if (iree_fseek64(stream->handle, offset, origin) == -1) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "failed file seek %d offset %" PRId64,
                              (int)seek_mode, offset);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_file_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  if (out_buffer_length) *out_buffer_length = 0;
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_pos_t read_length =
      fread(buffer, 1, buffer_capacity, stream->handle);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, read_length);

  iree_status_t status = iree_ok_status();
  if (read_length != buffer_capacity) {
    if (ferror(stream->handle)) {
      // Error during read.
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "failed to read up to %" PRIhsz
                                " bytes from the stream",
                                buffer_capacity);
    } else if (feof(stream->handle)) {
      // EOF encountered - not an error.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(eof)");
    }
  }

  if (out_buffer_length) *out_buffer_length = (iree_host_size_t)read_length;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_file_stream_write(iree_io_stream_t* base_stream,
                                               iree_host_size_t buffer_length,
                                               const void* buffer) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(buffer);
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (buffer_length) {
    size_t ret = fwrite(buffer, buffer_length, 1, stream->handle);
    if (ret != 1) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "unable to write %" PRIhsz " bytes to the stream", buffer_length);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_file_stream_fill(iree_io_stream_t* base_stream,
                                              iree_io_stream_pos_t count,
                                              const void* pattern,
                                              iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(base_stream);
  IREE_ASSERT_ARGUMENT(pattern);
  iree_io_file_stream_t* stream = iree_io_file_stream_cast(base_stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // This is very inefficient.
  for (iree_io_stream_pos_t i = 0; i < count; ++i) {
    if (fwrite(pattern, pattern_length, (size_t)count, stream->handle) !=
        count) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_make_status(
                  IREE_STATUS_DATA_LOSS,
                  "unable to write fill pattern to file at offset %" PRId64
                  " of %" PRId64 " total bytes",
                  (i * pattern_length), (count * pattern_length)));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_io_stream_vtable_t iree_io_file_stream_vtable = {
    .destroy = iree_io_file_stream_destroy,
    .offset = iree_io_file_stream_offset,
    .length = iree_io_file_stream_length,
    .seek = iree_io_file_stream_seek,
    .read = iree_io_file_stream_read,
    .write = iree_io_file_stream_write,
    .fill = iree_io_file_stream_fill,
    .map_read = NULL,
    .map_write = NULL,
};

#endif  // IREE_FILE_IO_ENABLE

//===----------------------------------------------------------------------===//
// Type wrappers and registration
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_io_stream, iree_io_stream_t);

#define IREE_VM_REGISTER_IO_STREAM_C_TYPE(instance, type, name, destroy_fn, \
                                          registration)                     \
  static const iree_vm_ref_type_descriptor_t registration##_storage = {     \
      .type_name = IREE_SVL(name),                                          \
      .offsetof_counter =                                                   \
          offsetof(type, ref_count) / IREE_VM_REF_COUNTER_ALIGNMENT,        \
      .destroy = (iree_vm_ref_destroy_t)destroy_fn,                         \
  };                                                                        \
  IREE_RETURN_IF_ERROR(iree_vm_instance_register_type(                      \
      instance, &registration##_storage, &registration));

IREE_API_EXPORT iree_status_t
iree_io_stream_module_register_types(iree_vm_instance_t* instance) {
  IREE_VM_REGISTER_IO_STREAM_C_TYPE(instance, iree_io_stream_t,
                                    "io_stream.handle", iree_io_stream_destroy,
                                    iree_io_stream_registration);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_io_stream_module_t {
  iree_allocator_t host_allocator;
  iree_io_stream_console_files_t console;
} iree_io_stream_module_t;

#define IREE_IO_STREAM_MODULE_CAST(module) \
  (iree_io_stream_module_t*)((uint8_t*)(module) + iree_vm_native_module_size())

typedef struct iree_io_stream_module_state_t {
  iree_allocator_t host_allocator;
  struct {
    iree_io_stream_t* stdin_stream;
    iree_io_stream_t* stdout_stream;
    iree_io_stream_t* stderr_stream;
  } console;
} iree_io_stream_module_state_t;

static void IREE_API_PTR iree_io_stream_module_destroy(void* base_module) {}

static void IREE_API_PTR iree_io_stream_module_free_state(
    void* self, iree_vm_module_state_t* module_state);

static iree_status_t IREE_API_PTR
iree_io_stream_module_alloc_state(void* self, iree_allocator_t host_allocator,
                                  iree_vm_module_state_t** out_module_state) {
  iree_io_stream_module_t* module = IREE_IO_STREAM_MODULE_CAST(self);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

#if IREE_FILE_IO_ENABLE
  if (iree_status_is_ok(status) && module->console.stdin_handle) {
    status = iree_io_file_stream_wrap(
        IREE_IO_STREAM_MODE_READABLE, module->console.stdin_handle,
        iree_io_stream_release_callback_null(), host_allocator,
        &state->console.stdin_stream);
  }
  if (iree_status_is_ok(status) && module->console.stdout_handle) {
    status = iree_io_file_stream_wrap(
        IREE_IO_STREAM_MODE_WRITABLE, module->console.stdout_handle,
        iree_io_stream_release_callback_null(), host_allocator,
        &state->console.stdout_stream);
  }
  if (iree_status_is_ok(status) && module->console.stderr_handle) {
    status = iree_io_file_stream_wrap(
        IREE_IO_STREAM_MODE_WRITABLE, module->console.stderr_handle,
        iree_io_stream_release_callback_null(), host_allocator,
        &state->console.stderr_stream);
  }
#endif  // IREE_FILE_IO_ENABLE

  if (iree_status_is_ok(status)) {
    *out_module_state = (iree_vm_module_state_t*)state;
  } else {
    iree_io_stream_module_free_state(module, (iree_vm_module_state_t*)state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void IREE_API_PTR iree_io_stream_module_free_state(
    void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_stream_module_state_t* state =
      (iree_io_stream_module_state_t*)module_state;
  iree_io_stream_release(state->console.stdin_stream);
  iree_io_stream_release(state->console.stdout_stream);
  iree_io_stream_release(state->console.stderr_stream);
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Casts a VM value to a C host size.
static iree_host_size_t iree_vm_cast_host_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if host
  // size is 32-bits.
  return (iree_host_size_t)value;
}

//===----------------------------------------------------------------------===//
// Exported functions
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_io_stream_module_console_stdin,  //
                   iree_io_stream_module_state_t,        //
                   v, r) {
  rets->r0 = iree_io_stream_retain_ref(state->console.stdin_stream);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_console_stdout,  //
                   iree_io_stream_module_state_t,         //
                   v, r) {
  rets->r0 = iree_io_stream_retain_ref(state->console.stdout_stream);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_console_stderr,  //
                   iree_io_stream_module_state_t,         //
                   v, r) {
  rets->r0 = iree_io_stream_retain_ref(state->console.stderr_stream);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_offset,   //
                   iree_io_stream_module_state_t,  //
                   r, I) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  rets->i0 = iree_io_stream_offset(stream);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_length,   //
                   iree_io_stream_module_state_t,  //
                   r, I) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  rets->i0 = iree_io_stream_length(stream);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_read_byte,  //
                   iree_io_stream_module_state_t,    //
                   r, i) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  uint8_t buffer[1] = {0};
  iree_host_size_t read_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_stream_read(stream, sizeof(buffer), buffer, &read_length));
  if (read_length != 1) {
    rets->i0 = -1;  // EOF
  } else {
    rets->i0 = (uint32_t)buffer[0];
  }
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_read_bytes,  //
                   iree_io_stream_module_state_t,     //
                   rrII, I) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &buffer));
  iree_host_size_t offset = iree_vm_cast_host_size(args->i2);
  iree_host_size_t length = iree_vm_cast_host_size(args->i3);
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_rw(buffer, offset, length, 1, &span));
  iree_host_size_t read_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_stream_read(stream, span.data_length, span.data, &read_length));
  rets->i0 = (int64_t)read_length;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_read_delimiter,  //
                   iree_io_stream_module_state_t,         //
                   ri, r) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  uint8_t delimiter = (uint8_t)args->i1;

  // Growable heap storage for the string as it is read.
  iree_host_size_t capacity = 128;
  iree_host_size_t length = 0;
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(state->host_allocator, capacity, (void**)&storage));

  iree_status_t status = iree_ok_status();
  bool is_eos = false;
  do {
    // Read the next character.
    uint8_t char_buffer[1] = {0};
    iree_host_size_t read_length = 0;
    status = iree_io_stream_read(stream, sizeof(char_buffer), char_buffer,
                                 &read_length);
    if (!iree_status_is_ok(status)) break;

    // If the end of the stream was hit or the delimiter was parsed then bail.
    if (read_length < sizeof(char_buffer)) {
      is_eos = true;
      break;
    } else if (char_buffer[0] == delimiter) {
      --read_length;
      break;
    }

    // Append character to the storage buffer.
    if (length + 1 >= capacity) {
      iree_host_size_t new_capacity = capacity * 2;
      status = iree_allocator_realloc(state->host_allocator, new_capacity,
                                      (void**)&storage);
      if (!iree_status_is_ok(status)) break;
      capacity = new_capacity;
    }
    memcpy(&storage[length], char_buffer, sizeof(char_buffer));
    length += sizeof(char_buffer);
  } while (length < IREE_HOST_SIZE_MAX);

  // Substring the valid bytes.
  iree_vm_buffer_t* string = NULL;
  if (iree_status_is_ok(status) && !is_eos) {
    status = iree_vm_buffer_create(
        IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST | IREE_VM_BUFFER_ACCESS_MUTABLE,
        length, 1, state->host_allocator, &string);
  }
  if (iree_status_is_ok(status) && string && length) {
    status = iree_vm_buffer_write_elements(storage, string, 0, length, 1);
  }
  iree_allocator_free(state->host_allocator, storage);

  if (iree_status_is_ok(status)) {
    rets->r0 = iree_vm_buffer_move_ref(string);
  } else {
    iree_vm_buffer_release(string);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_write_byte,  //
                   iree_io_stream_module_state_t,     //
                   ri, v) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  uint8_t buffer[1] = {(uint8_t)args->i1};
  return iree_io_stream_write(stream, 1, buffer);
}

IREE_VM_ABI_EXPORT(iree_io_stream_module_write_bytes,  //
                   iree_io_stream_module_state_t,      //
                   rrII, v) {
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_stream_check_deref(args->r0, &stream));
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &buffer));
  iree_host_size_t offset = iree_vm_cast_host_size(args->i2);
  iree_host_size_t length = iree_vm_cast_host_size(args->i3);
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(buffer, offset, length, 1, &span));
  return iree_io_stream_write(stream, span.data_length, span.data);
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_io_stream_module_exports_
// table.
static const iree_vm_native_function_ptr_t iree_io_stream_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/io/stream/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t
    iree_io_stream_module_imports_[1];

static const iree_vm_native_export_descriptor_t
    iree_io_stream_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "iree/modules/io/stream/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_io_stream_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_io_stream_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t
    iree_io_stream_module_descriptor_ = {
        .name = iree_string_view_literal("io_stream"),
        .version = IREE_IO_STREAM_MODULE_VERSION_LATEST,
        .attr_count = 0,
        .attrs = NULL,
        .dependency_count = 0,
        .dependencies = NULL,
        .import_count = 0,  // workaround for 0-length C struct
        .imports = iree_io_stream_module_imports_,
        .export_count = IREE_ARRAYSIZE(iree_io_stream_module_exports_),
        .exports = iree_io_stream_module_exports_,
        .function_count = IREE_ARRAYSIZE(iree_io_stream_module_funcs_),
        .functions = iree_io_stream_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_io_stream_module_create(
    iree_vm_instance_t* instance, iree_io_stream_console_files_t console,
    iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_io_stream_module_destroy,
      .alloc_state = iree_io_stream_module_alloc_state,
      .free_state = iree_io_stream_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_io_stream_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_io_stream_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_io_stream_module_t* module = IREE_IO_STREAM_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->console = console;

  *out_module = base_module;
  return iree_ok_status();
}
