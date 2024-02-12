// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/numpy_io.h"

//===----------------------------------------------------------------------===//
// .npy (multiple values concatenated)
//===----------------------------------------------------------------------===//

// File format spec:
// https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
//
// 6b: magic `\x93NUMPY`
// 1b: major version
// 1b: minor version
// if major version == 1:
//   2b: length of the header
// elif major version > 1:
//   4b: length of the header
// [header length]b ascii/utf8 newline-terminated string
//   padded with spaces (\x20) such that
//   `len(magic string) + 2 + len(length) + HEADER_LEN` % 64 = 0

// Reads the numpy file header string into an allocated |out_header_buffer|.
// Upon successful return the |stream| will be positioned immediately at the
// start of the file payload.
// |out_header_buffer| must be freed by the caller with |host_allocator|.
static iree_status_t iree_numpy_npy_read_header(
    iree_io_stream_t* stream, iree_allocator_t host_allocator,
    iree_host_size_t* out_header_length, char** out_header_buffer) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_header_length);
  IREE_ASSERT_ARGUMENT(out_header_buffer);
  *out_header_length = 0;
  *out_header_buffer = NULL;

  // Since the header contents vary based on version we read the fixed prefix
  // first and then continue with the rest.
  struct {
    uint8_t magic[6];
    uint8_t version_major;
    uint8_t version_minor;
  } header;
  static_assert(sizeof(header) == 8, "packing");
  IREE_RETURN_IF_ERROR(
      iree_io_stream_read(stream, sizeof(header), &header, NULL),
      "unable to read entire header prefix");

  // Verify magic bytes to confirm this is an npy file.
  static const uint8_t kMagicBytes[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (memcmp(header.magic, kMagicBytes, sizeof(kMagicBytes)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "npy header magic mismatch");
  }

  // Ensure we support the version; newer versions aren't expected to parse.
  // There's been no minor versions yet so we only need to check major.
  if (header.version_major <= 0 || header.version_major > 3) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "npy version %d.%d not supported",
                            header.version_major, header.version_minor);
  }

  // Read 2- or 4-byte header length.
  // Have never seen a header actually needing 4-bytes (any reason to have one
  // wouldn't matter here as we only need dtype and shape), but the v3.x format
  // always uses 4-byte headers. numpy still saves in the v1.x 2-byte format for
  // compatibility so we do the same.
  iree_host_size_t header_length = 0;
  if (header.version_major == 1) {
    uint16_t header_length_u16 = 0;
    IREE_RETURN_IF_ERROR(iree_io_stream_read(stream, sizeof(header_length_u16),
                                             &header_length_u16, NULL),
                         "failed to read version %d.%d 2-byte header length",
                         header.version_major, header.version_minor);
    header_length = header_length_u16;
  } else {
    uint32_t header_length_u32 = 0;
    IREE_RETURN_IF_ERROR(iree_io_stream_read(stream, sizeof(header_length_u32),
                                             &header_length_u32, NULL),
                         "failed to read version %d.%d 4-byte header length",
                         header.version_major, header.version_minor);
    header_length = header_length_u32;
  }

  // Allocate storage for the header string - the caller will need to free it.
  // We could fscanf, but this way we can reuse all our string view utilities.
  char* header_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, header_length,
                                             (void**)&header_buffer));

  // Read entire header string, including padding/newline.
  iree_status_t status = iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_read(stream, header_length, header_buffer, NULL),
      "failed to read header string of %" PRIhsz " bytes", header_length);

  if (iree_status_is_ok(status)) {
    // Caller must free the string buffer.
    *out_header_length = header_length;
    *out_header_buffer = header_buffer;
  } else {
    iree_allocator_free(host_allocator, header_buffer);
  }
  return status;
}

typedef struct {
  iree_io_stream_t* stream;
} iree_numpy_npy_read_params_t;
static iree_status_t iree_numpy_npy_read_into_mapping(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_numpy_npy_read_params_t* params =
      (iree_numpy_npy_read_params_t*)user_data;
  IREE_RETURN_IF_ERROR(
      iree_io_stream_read(params->stream, mapping->contents.data_length,
                          mapping->contents.data, NULL),
      "failed to read npy contents of %" PRIhsz " bytes",
      mapping->contents.data_length);
  return iree_ok_status();
}

// Scans for the next key: value pair in |dict|.
// |dict| will be set to the remaining |dict| string after the key and value.
static iree_status_t iree_numpy_consume_dict_key_value(
    iree_string_view_t* dict, iree_string_view_t* out_key,
    iree_string_view_t* out_value) {
  // Split `'key':` from the remainder of the string.
  if (iree_string_view_split(*dict, ':', out_key, dict) == -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed header dict");
  }
  iree_string_view_consume_prefix(out_key, IREE_SV("'"));
  iree_string_view_consume_suffix(out_key, IREE_SV("'"));

  // Scan to the end of the value. As dict pairs are comma delimited and we
  // have commas inside of the values we need to check for nesting.
  // We should have something remaining like:
  //   ` False, ...`
  //   ` '|i4', ...`
  //   ` (1,), ...`
  //   ` (1, 2), ...`
  // To make things simpler we just check for these cases.
  *dict = iree_string_view_trim(*dict);
  if (iree_string_view_consume_prefix(dict, IREE_SV("True"))) {
    *out_value = IREE_SV("True");
  } else if (iree_string_view_consume_prefix(dict, IREE_SV("False"))) {
    *out_value = IREE_SV("False");
  } else if (iree_string_view_consume_prefix(dict, IREE_SV("'"))) {
    iree_string_view_split(*dict, '\'', out_value, dict);
  } else if (iree_string_view_consume_prefix(dict, IREE_SV("("))) {
    iree_string_view_split(*dict, ')', out_value, dict);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed header dict");
  }
  if (!iree_string_view_consume_prefix(dict, IREE_SV(","))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed header dict");
  }
  *dict = iree_string_view_trim(*dict);
  return iree_ok_status();
}

// Returns true if the byte order is little-endian (or :shrug:).
// Consumes the prefix character if present.
// https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
static bool iree_numpy_consume_descr_byte_order(iree_string_view_t* descr) {
  if (descr->size == 0) return false;
  char c = descr->data[0];
  if (c == '|' || c == '=' || c == '<') {
    // Little-endian (or native, which is little).
    *descr = iree_string_view_remove_prefix(*descr, 1);
    return true;
  } else if (c == '>') {
    // Big-endian.
    return false;
  } else {
    // No byte order marker.
    return true;
  }
}

// Parses the |descr| string into an |out_element_type|.
// Returns failure if the descr is not supported.
// https://numpy.org/doc/stable/reference/arrays.dtypes.html
static iree_status_t iree_numpy_descr_to_element_type(
    iree_string_view_t descr, iree_hal_element_type_t* out_element_type) {
  // Check endianness prefix indicates little-endian (or default).
  if (!iree_numpy_consume_descr_byte_order(&descr)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "big-endian data type '%.*s' unsupported",
                            (int)descr.size, descr.data);
  }
  if (iree_string_view_is_empty(descr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "descr missing kind");
  }

  // First char is kind and the remaining should be the byte width.
  // https://numpy.org/doc/stable/reference/arrays.interface.html#arrays-interface
  char kind = descr.data[0];
  descr = iree_string_view_remove_prefix(descr, 1);
  iree_hal_numerical_type_t numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
  switch (kind) {
    case 'b':  // boolean
      numerical_type = IREE_HAL_NUMERICAL_TYPE_BOOLEAN;
      break;
    case 'i':  // (signed) integer
      numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED;
      break;
    case 'u':  // unsigned integer
      numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED;
      break;
    case 'f':  // floating-point
      numerical_type = IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE;
      break;
    case 'c':  // complex-floating point
      numerical_type = IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX;
      break;
    case 'V':  // raw data
      numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
      break;
    default:
    case 'm':  // timedelta
    case 'M':  // datetime
    case 'O':  // python objects
    case 'U':  // unicode string (count is number of characters)
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported data type %c", kind);
  }

  // Parse the byte width.
  uint32_t byte_width = 0;
  if (!iree_string_view_atoi_uint32(descr, &byte_width)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid descr byte width");
  }

  *out_element_type =
      IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, byte_width * 8);
  return iree_ok_status();
}

// Counts the number of dimensions in a |shape| string.
// Examples:
//   `` = 0
//   `1,` = 1
//   `2, 2, 1` = 3
static iree_host_size_t iree_numpy_parse_shape_rank(iree_string_view_t shape) {
  if (iree_string_view_is_empty(shape)) return 0;
  // NOTE: possibility of trailing , because python.
  iree_string_view_consume_suffix(&shape, IREE_SV(","));
  iree_host_size_t rank = 1;
  for (iree_host_size_t i = 0; i < shape.size; ++i) {
    if (shape.data[i] == ',') ++rank;
  }
  return rank;
}

// Parses shape dimensions from a |shape| string like `(2, 2, 1)`.
// |shape_rank| must match that returned by iree_numpy_parse_shape_rank
// and |out_shape| must have storage for at least as many elements.
static iree_status_t iree_numpy_parse_shape_dims(iree_string_view_t shape,
                                                 iree_host_size_t shape_rank,
                                                 iree_hal_dim_t* out_shape) {
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    iree_string_view_t dim_str;
    iree_string_view_split(shape, ',', &dim_str, &shape);
    dim_str = iree_string_view_trim(dim_str);
    uint64_t dim = 0;
    if (!iree_string_view_atoi_uint64(dim_str, &dim)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid dimension %" PRIhsz " value %.*s", i,
                              (int)dim_str.size, dim_str.data);
    }
    out_shape[i] = (iree_hal_dim_t)dim;
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_numpy_npy_load_ndarray(
    iree_io_stream_t* stream, iree_numpy_npy_load_options_t options,
    iree_hal_buffer_params_t buffer_params, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  *out_buffer_view = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(device_allocator);

  // Quick check for EOF; if already there we can give a better error than
  // if we failed trying to parse the header. Since npy files are often
  // concatenated callers are likely to be using this in a loop and checking for
  // this condition, even if it'd be better if they did it themselves.
  if (iree_io_stream_is_eos(stream)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "end-of-file");
  }

  // Read header string.
  // The resulting header must be freed with host_allocator.
  char* header_buffer = NULL;
  iree_host_size_t header_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_numpy_npy_read_header(stream, host_allocator, &header_length,
                                     &header_buffer));
  iree_string_view_t header = iree_string_view_trim(
      iree_make_string_view(header_buffer, header_length));

  // Parse the header.
  // It look something like this:
  //   {'descr': '|i1', 'fortran_order': False, 'shape': (2, 2, 1), }
  // The spec says that although the keys should be sorted alphabetically that's
  // not a requirement (yuck) and we have to handle out-of-order keys. There may
  // also be keys we don't understand such as when what's saved is a pickled
  // object. We implement a basic scanning parser here and try to deal with it.
  iree_status_t status = iree_ok_status();
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t* shape = NULL;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_hal_encoding_type_t encoding_type = IREE_HAL_ENCODING_TYPE_OPAQUE;
  iree_string_view_consume_prefix(&header, IREE_SV("{"));
  iree_string_view_consume_suffix(&header, IREE_SV("}"));
  while (!iree_string_view_is_empty(header)) {
    // header => 'key': value{, header}
    iree_string_view_t key, value;
    status = iree_numpy_consume_dict_key_value(&header, &key, &value);
    if (!iree_status_is_ok(status)) break;

    if (iree_string_view_equal(key, IREE_SV("descr"))) {
      status = iree_numpy_descr_to_element_type(value, &element_type);
    } else if (iree_string_view_equal(key, IREE_SV("fortran_order"))) {
      if (iree_string_view_equal(value, IREE_SV("False"))) {
        encoding_type = IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;
      } else {
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "fortran order arrays not supported");
      }
    } else if (iree_string_view_equal(key, IREE_SV("shape"))) {
      shape_rank = iree_numpy_parse_shape_rank(value);
      if (shape_rank > 128) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "shape rank %" PRIhsz
                                  " too large; be reasonable please");
      } else {
        shape = iree_alloca(shape_rank * sizeof(*shape));
        status = iree_numpy_parse_shape_dims(value, shape_rank, shape);
      }
    }
    if (!iree_status_is_ok(status)) break;
  }

  // Allocate the buffer view and directly read into the allocated memory.
  // On targets where we can perform host mapping this will be zero-copy; on
  // others it'll at least be _somewhat_ efficient.
  if (iree_status_is_ok(status)) {
    iree_numpy_npy_read_params_t read_params = {
        .stream = stream,
    };
    buffer_params.access |= IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;
    status = iree_hal_buffer_view_generate_buffer(
        device, device_allocator, shape_rank, shape, element_type,
        encoding_type, buffer_params, iree_numpy_npy_read_into_mapping,
        &read_params, out_buffer_view);
  }

  iree_allocator_free(host_allocator, header_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Builds a dtype string from |buffer_view|.
static iree_status_t iree_numpy_npy_build_dtype(
    iree_hal_buffer_view_t* buffer_view, iree_string_builder_t* builder) {
  iree_hal_element_type_t element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  iree_hal_numerical_type_t numerical_type =
      iree_hal_element_numerical_type(element_type);
  iree_host_size_t byte_count = iree_hal_element_dense_byte_count(element_type);

  // Always little-endian, but 1 byte elements don't apply.
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, byte_count == 1 ? "|" : "<"));

  // Type prefix.
  switch (numerical_type) {
    case IREE_HAL_NUMERICAL_TYPE_UNKNOWN: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "V"));
      break;
    }
    case IREE_HAL_NUMERICAL_TYPE_BOOLEAN: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "b"));
      break;
    }
    case IREE_HAL_NUMERICAL_TYPE_INTEGER:
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "i"));
      break;
    }
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "u"));
      break;
    }
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "f"));
      break;
    }
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX: {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "c"));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported data encoding");
  }

  // Width in bytes.
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "%" PRIhsz, byte_count));

  return iree_ok_status();
}

// Builds a shape string from |buffer_view|.
// Examples:
//   `` (rank 0)
//   `1,` (rank 1)
//   `2, 2, 1` (rank 3)
static iree_status_t iree_numpy_npy_build_shape(
    iree_hal_buffer_view_t* buffer_view, iree_string_builder_t* builder) {
  iree_host_size_t shape_rank = iree_hal_buffer_view_shape_rank(buffer_view);
  if (shape_rank == 0) return iree_ok_status();

  // dim, dim, ...
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    iree_hal_dim_t dim = iree_hal_buffer_view_shape_dim(buffer_view, i);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIdim, dim));
  }

  // Trailing , needed for rank 1 because python.
  if (shape_rank == 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  }
  return iree_ok_status();
}

// Builds a header dict string in |builder| for the given |buffer_view|.
static iree_status_t iree_numpy_npy_build_header(
    iree_numpy_npy_save_options_t options, iree_hal_buffer_view_t* buffer_view,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));

  // 'descr': '{type}'
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "'descr': '"));
  IREE_RETURN_IF_ERROR(iree_numpy_npy_build_dtype(buffer_view, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "', "));

  // 'fortran_order': False
  iree_hal_encoding_type_t encoding_type =
      iree_hal_buffer_view_encoding_type(buffer_view);
  if (encoding_type != IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-row-major contiguous encoding not supported for serialization");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "'fortran_order': False, "));

  // 'shape': ({shape})
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "'shape': ("));
  IREE_RETURN_IF_ERROR(iree_numpy_npy_build_shape(buffer_view, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "), "));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  return iree_ok_status();
}

// Writes the numpy file |header_dict| and pads the file to the required
// alignment. |header_dict| should not have any trailing padding or the newline
// character.
static iree_status_t iree_numpy_npy_write_header(
    iree_io_stream_t* stream, iree_numpy_npy_save_options_t options,
    iree_string_view_t header_dict) {
  // v1 -> v2 if the header requires it; we don't but good to be conformant.
  bool requires_v2 = header_dict.size > 65535;

  // Write the header.
  // We only use version 1 or 2 based on the header length for maximum
  // compatibility (same as what numpy does).
  struct {
    uint8_t magic[6];
    uint8_t version_major;
    uint8_t version_minor;
  } header = {
      .magic = {0x93, 'N', 'U', 'M', 'P', 'Y'},
      .version_major = requires_v2 ? 2 : 1,
      .version_minor = 0,
  };
  static_assert(sizeof(header) == 8, "padding");
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header),
                       "failed to write header prefix");

  // Pad out what we write to 64b.
  // Note that this includes the header prefix, length, dict, and newline.
  iree_host_size_t current_length =
      sizeof(header) + (requires_v2 ? 4 : 2) + header_dict.size + /*\n*/ 1;
  iree_host_size_t padded_length = iree_host_align(current_length, 64);
  iree_host_size_t padding_length = padded_length - current_length;

  // Write header length.
  iree_host_size_t header_length = header_dict.size + padding_length + /*\n*/ 1;
  if (requires_v2) {
    uint32_t header_length_u32 = (uint32_t)header_length;
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header_length_u32),
                                              &header_length_u32),
                         "failed to write header length");
  } else {
    uint16_t header_length_u16 = (uint16_t)header_length;
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header_length_u16),
                                              &header_length_u16),
                         "failed to write header length");
  }

  // Write header contents (without padding/trailing newline).
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(stream, header_dict.size, header_dict.data),
      "failed to write header contents");

  // Add space padding up to 64b alignment (minus newline).
  const char space = ' ';
  IREE_RETURN_IF_ERROR(
      iree_io_stream_fill(stream, padding_length, &space, sizeof(space)),
      "failed to pad header");

  // Trailing newline, which should put us right at the %64=0 alignment.
  IREE_RETURN_IF_ERROR(iree_io_stream_write_char(stream, '\n'),
                       "failed to write trailing newline");

  return iree_ok_status();
}

// Writes |buffer_view| contents to |stream|.
static iree_status_t iree_numpy_npy_write_bytes(
    iree_io_stream_t* stream, iree_hal_buffer_view_t* buffer_view) {
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_device_size_t write_length =
      iree_hal_buffer_view_byte_length(buffer_view);

  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      write_length, &mapping));

  iree_status_t status = iree_status_annotate(
      iree_io_stream_write(stream, write_length, mapping.contents.data),
      IREE_SV("failed to write buffer contents"));

  IREE_IGNORE_ERROR(iree_hal_buffer_unmap_range(&mapping));
  return status;
}

IREE_API_EXPORT iree_status_t iree_numpy_npy_save_ndarray(
    iree_io_stream_t* stream, iree_numpy_npy_save_options_t options,
    iree_hal_buffer_view_t* buffer_view, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build header dict string.
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_numpy_npy_build_header(options, buffer_view, &builder);

  // Write header magic and dict, padded to 64 bytes.
  if (iree_status_is_ok(status)) {
    status = iree_numpy_npy_write_header(stream, options,
                                         iree_string_builder_view(&builder));
  }

  // Write buffer contents.
  if (iree_status_is_ok(status)) {
    status = iree_numpy_npy_write_bytes(stream, buffer_view);
  }

  iree_string_builder_deinitialize(&builder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
