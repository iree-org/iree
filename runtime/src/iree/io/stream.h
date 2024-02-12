// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_STREAM_H_
#define IREE_IO_STREAM_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_stream_t
//===----------------------------------------------------------------------===//

// Stream position; absolute or relative based on how it's used.
typedef int64_t iree_io_stream_pos_t;

// Bits defining which operations are allowed on a stream.
enum iree_io_stream_mode_bits_t {
  // Allows operations that read from the stream.
  IREE_IO_STREAM_MODE_READABLE = 1u << 0,
  // Allows operations that write to the stream.
  IREE_IO_STREAM_MODE_WRITABLE = 1u << 1,
  // Allows the iree_io_stream_resize operation for extend/truncate.
  IREE_IO_STREAM_MODE_RESIZABLE = 1u << 2,
  // Allows the iree_io_stream_seek operation.
  IREE_IO_STREAM_MODE_SEEKABLE = 1u << 3,
  // Allows iree_io_stream_map_read/iree_io_stream_map_write operations.
  IREE_IO_STREAM_MODE_MAPPABLE = 1u << 4,
};
typedef uint32_t iree_io_stream_mode_t;

typedef enum iree_io_stream_seek_mode_t {
  // Like fseek(SEEK_SET): seek to an absolute offset in the stream.
  // A value of 0 indicates start-of-stream and a value of the total stream
  // length indicates end-of-stream.
  IREE_IO_STREAM_SEEK_SET = 0,
  // Like fseek(SEEK_CUR): seek is a relative offset from the current stream
  // offset, positive or negative. A value of 0 indicates no change.
  IREE_IO_STREAM_SEEK_FROM_CURRENT,
  // Like fseek(SEEK_END): seek is a negative offset from the end of the stream.
  // A value of 0 indicates end-of-stream (one byte past the end of the stream,
  // with the offset equalling the total stream length).
  //
  // Example:
  //   0123456789
  //           ^^^
  //           ||+- SEEK_FROM_END  0
  //           |+-- SEEK_FROM_END -1
  //           +--- SEEK_FROM_END -2
  IREE_IO_STREAM_SEEK_FROM_END,
} iree_io_stream_seek_mode_t;

typedef struct iree_io_stream_t iree_io_stream_t;

// Retains the given |stream| for the caller.
IREE_API_EXPORT void iree_io_stream_retain(iree_io_stream_t* stream);

// Releases the given |stream| from the caller.
IREE_API_EXPORT void iree_io_stream_release(iree_io_stream_t* stream);

// Returns the mode of the stream indicating what operations may be performed.
IREE_API_EXPORT iree_io_stream_mode_t
iree_io_stream_mode(const iree_io_stream_t* stream);

// Returns the position of the stream relative to the stream base offset.
IREE_API_EXPORT iree_io_stream_pos_t
iree_io_stream_offset(iree_io_stream_t* stream);

// Returns the total length of the stream in bytes ignoring the current offset.
IREE_API_EXPORT iree_io_stream_pos_t
iree_io_stream_length(iree_io_stream_t* stream);

// Returns true if |stream| is positioned at the end of the stream.
// When at the end of stream reads will fail and writes will append.
IREE_API_EXPORT bool iree_io_stream_is_eos(iree_io_stream_t* stream);

// Seeks within |stream| to |seek_offset| based on the given |seek_mode|.
// If IREE_IO_STREAM_MODE_SEEKABLE is not set then only forward relative seeks
// are supported.
IREE_API_EXPORT iree_status_t iree_io_stream_seek(
    iree_io_stream_t* stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset);

// Seeks within |stream| to the next offset with the specified |alignment|.
// The alignment is expected to be a power-of-two value.
// No-op if the stream offset is already aligned. Valid even if
// IREE_IO_STREAM_MODE_SEEKABLE is not set as this only performs a forward
// relative seek.
IREE_API_EXPORT iree_status_t iree_io_stream_seek_to_alignment(
    iree_io_stream_t* stream, iree_io_stream_pos_t alignment);

// Reads up to |buffer_capacity| bytes from |stream| into |buffer|.
// Returns the total number of bytes read in the optional |out_buffer_length|,
// which may be less than the capacity if the end of stream is reached. If
// |out_buffer_length| is not provided the read will fail if enough bytes are
// not available. Requires the stream have IREE_IO_STREAM_MODE_READABLE.
IREE_API_EXPORT iree_status_t
iree_io_stream_read(iree_io_stream_t* stream, iree_host_size_t buffer_capacity,
                    void* buffer, iree_host_size_t* out_buffer_length);

// Writes |buffer_length| bytes from |buffer| to |stream|.
// Requires the stream have IREE_IO_STREAM_MODE_WRITABLE.
IREE_API_EXPORT iree_status_t
iree_io_stream_write(iree_io_stream_t* stream, iree_host_size_t buffer_length,
                     const void* buffer);

// Writes a single character/byte to the stream.
// Requires the stream have IREE_IO_STREAM_MODE_WRITABLE.
IREE_API_EXPORT iree_status_t
iree_io_stream_write_char(iree_io_stream_t* stream, char c);

// Writes a string view to the stream (excluding NUL terminator).
// Requires the stream have IREE_IO_STREAM_MODE_WRITABLE.
IREE_API_EXPORT iree_status_t
iree_io_stream_write_string(iree_io_stream_t* stream, iree_string_view_t value);

// Writes |count| elements of |pattern_length| with the given |pattern| value.
// Requires the stream have IREE_IO_STREAM_MODE_WRITABLE.
IREE_API_EXPORT iree_status_t
iree_io_stream_fill(iree_io_stream_t* stream, iree_io_stream_pos_t count,
                    const void* pattern, iree_host_size_t pattern_length);

// Maps a span of |length| bytes for reading. The stream offset is advanced by
// |length| immediately and the caller can read the returned |out_span| contents
// prior to closing the file.
// Requires that IREE_IO_STREAM_MODE_READABLE and IREE_IO_STREAM_MODE_MAPPABLE
// are set.
IREE_API_EXPORT iree_status_t
iree_io_stream_map_read(iree_io_stream_t* stream, iree_host_size_t length,
                        iree_const_byte_span_t* out_span);

// Maps a span of |length| bytes for writing. The stream offset is advanced by
// |length| immediately and the caller must fill the returned |out_span| with
// contents prior to closing the file.
// Requires that IREE_IO_STREAM_MODE_WRITABLE and IREE_IO_STREAM_MODE_MAPPABLE
// are set.
IREE_API_EXPORT iree_status_t
iree_io_stream_map_write(iree_io_stream_t* stream, iree_host_size_t length,
                         iree_byte_span_t* out_span);

// Copies |length| bytes from |source_stream| to |target_stream|.
// Requires |source_stream| have IREE_IO_STREAM_MODE_READABLE and
// |target_stream| have IREE_IO_STREAM_MODE_WRITABLE.
IREE_API_EXPORT iree_status_t iree_io_stream_copy(
    iree_io_stream_t* source_stream, iree_io_stream_t* target_stream,
    iree_io_stream_pos_t length);

//===----------------------------------------------------------------------===//
// iree_io_stream_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_io_stream_vtable_t {
  void(IREE_API_PTR* destroy)(iree_io_stream_t* IREE_RESTRICT stream);

  iree_io_stream_pos_t(IREE_API_PTR* offset)(iree_io_stream_t* stream);
  iree_io_stream_pos_t(IREE_API_PTR* length)(iree_io_stream_t* stream);
  iree_status_t(IREE_API_PTR* seek)(iree_io_stream_t* stream,
                                    iree_io_stream_seek_mode_t seek_mode,
                                    iree_io_stream_pos_t seek_offset);
  iree_status_t(IREE_API_PTR* read)(iree_io_stream_t* stream,
                                    iree_host_size_t buffer_capacity,
                                    void* buffer,
                                    iree_host_size_t* out_buffer_length);
  iree_status_t(IREE_API_PTR* write)(iree_io_stream_t* stream,
                                     iree_host_size_t buffer_length,
                                     const void* buffer);
  iree_status_t(IREE_API_PTR* fill)(iree_io_stream_t* stream,
                                    iree_io_stream_pos_t count,
                                    const void* pattern,
                                    iree_host_size_t pattern_length);
  iree_status_t(IREE_API_PTR* map_read)(iree_io_stream_t* stream,
                                        iree_host_size_t length,
                                        iree_const_byte_span_t* out_span);
  iree_status_t(IREE_API_PTR* map_write)(iree_io_stream_t* stream,
                                         iree_host_size_t length,
                                         iree_byte_span_t* out_span);
} iree_io_stream_vtable_t;

struct iree_io_stream_t {
  iree_atomic_ref_count_t ref_count;
  const iree_io_stream_vtable_t* vtable;
  iree_io_stream_mode_t mode;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_STREAM_H_
