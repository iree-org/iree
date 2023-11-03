// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/gguf/gguf_format.h"

#include <ctype.h>

// File format:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// References:
// https://github.com/ggerganov/ggml/blob/master/src/ggml.c
// https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf.py
//
// Unfortunately they don't encode tensor sizes so we need to carry a ton of
// logic/tables for calculating it. We could avoid this by just saying every
// tensor is the remaining length in the file but that'd be really confusing.
//
// Things that would improve the format:
// - alignment of all fields in the header so that they can be directly accessed
//   (since strings and such are stored inline and have byte-length alignment
//   a single string early in the header ensures all other fields in all other
//   header data are unaligned, preventing us from directly overlaying structs)
// - all variable-length header data should be in tables as with most other
//   formats designed for easy parsing/high performance I/O (ELF, etc) - strings
//   and dimensions as well as embedded metadata arrays should be offsets to
//   the data in the file instead of inlined
// - tensor total size should be stored to allow for quickly partitioning files
//   without relying on the various data type/block structures (some of which
//   are only conditionally available and will change frequently)
// - header total size (offset to tensor_data) should be stored to allow
//   skipping the header or quickly checking for truncated files (currently need
//   to correctly parse the entire header in order to find the tensor_data)
// - metadata arrays should have a total length to make it easy to skip them;
//   since they can contain strings it's not possible given an array type and
//   element count to know how long it is without parsing all elements and
//   arrays of strings are even worse - fixed size header fields with externally
//   referenced variable-length data would make this easy
//
// The structs below are here for reference; some we use with slight
// modifications for readability (easy to compare against reference code) but
// others we just parse field-by-field due to the aforementioned nested
// variable-sized craziness.

#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

enum ggml_type_e {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_I8 = 16,
  GGML_TYPE_I16 = 17,
  GGML_TYPE_I32 = 18,
  GGML_TYPE_COUNT,
};
typedef uint32_t ggml_type_t;

#define QK4_0 32
typedef struct {
  uint16_t d;
  uint8_t qs[QK4_0 / 2];
} block_q4_0;
#define QK4_1 32
typedef struct {
  uint16_t d;
  uint16_t m;
  uint8_t qs[QK4_1 / 2];
} block_q4_1;
#define QK5_0 32
typedef struct {
  uint16_t d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
} block_q5_0;
#define QK5_1 32
typedef struct {
  uint16_t d;
  uint16_t m;
  uint8_t qh[4];
  uint8_t qs[QK5_1 / 2];
} block_q5_1;
#define QK8_0 32
typedef struct {
  uint16_t d;
  int8_t qs[QK8_0];
} block_q8_0;
#define QK8_1 32
typedef struct {
  float d;
  float s;
  int8_t qs[QK8_1];
} block_q8_1;

typedef struct {
  int blck_size;
  size_t type_size;
} ggml_type_traits_t;
static const ggml_type_traits_t ggml_type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] =
        {
            .blck_size = 1,
            .type_size = sizeof(int8_t),
        },
    [GGML_TYPE_I16] =
        {
            .blck_size = 1,
            .type_size = sizeof(int16_t),
        },
    [GGML_TYPE_I32] =
        {
            .blck_size = 1,
            .type_size = sizeof(int32_t),
        },
    [GGML_TYPE_F32] =
        {
            .blck_size = 1,
            .type_size = sizeof(float),
        },
    [GGML_TYPE_F16] =
        {
            .blck_size = 1,
            .type_size = sizeof(uint16_t),
        },
    [GGML_TYPE_Q4_0] =
        {
            .blck_size = QK4_0,
            .type_size = sizeof(block_q4_0),
        },
    [GGML_TYPE_Q4_1] =
        {
            .blck_size = QK4_1,
            .type_size = sizeof(block_q4_1),
        },
    [GGML_TYPE_Q5_0] =
        {
            .blck_size = QK5_0,
            .type_size = sizeof(block_q5_0),
        },
    [GGML_TYPE_Q5_1] =
        {
            .blck_size = QK5_1,
            .type_size = sizeof(block_q5_1),
        },
    [GGML_TYPE_Q8_0] =
        {
            .blck_size = QK8_0,
            .type_size = sizeof(block_q8_0),
        },
    [GGML_TYPE_Q8_1] =
        {
            .blck_size = QK8_1,
            .type_size = sizeof(block_q8_1),
        },
};

enum gguf_metadata_value_type_e {
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};
typedef uint32_t gguf_metadata_value_type_t;
static const iree_host_size_t gguf_metadata_value_type_sizes[] = {
    [GGUF_METADATA_VALUE_TYPE_UINT8] = sizeof(uint8_t),
    [GGUF_METADATA_VALUE_TYPE_INT8] = sizeof(int8_t),
    [GGUF_METADATA_VALUE_TYPE_UINT16] = sizeof(uint16_t),
    [GGUF_METADATA_VALUE_TYPE_INT16] = sizeof(int16_t),
    [GGUF_METADATA_VALUE_TYPE_UINT32] = sizeof(uint32_t),
    [GGUF_METADATA_VALUE_TYPE_INT32] = sizeof(int32_t),
    [GGUF_METADATA_VALUE_TYPE_FLOAT32] = sizeof(float),
    [GGUF_METADATA_VALUE_TYPE_BOOL] = sizeof(bool),
    [GGUF_METADATA_VALUE_TYPE_STRING] = 0,
    [GGUF_METADATA_VALUE_TYPE_ARRAY] = 0,
    [GGUF_METADATA_VALUE_TYPE_UINT64] = sizeof(uint64_t),
    [GGUF_METADATA_VALUE_TYPE_INT64] = sizeof(int64_t),
    [GGUF_METADATA_VALUE_TYPE_FLOAT64] = sizeof(double),
};

// typedef struct {
//   uint64_t len;
//   char string[len];
// } gguf_string_t;

// NOTE: storage of string/value has interior variable length data (ew).
// union gguf_metadata_value_t {
//   uint8_t uint8;
//   int8_t int8;
//   uint16_t uint16;
//   int16_t int16;
//   uint32_t uint32;
//   int32_t int32;
//   float float32;
//   uint64_t uint64;
//   int64_t int64;
//   double float64;
//   bool bool_;
//   gguf_string_t string;
//   struct {
//     gguf_metadata_value_type_t type;
//     uint64_t len;
//     gguf_metadata_value_t array[len];
//   } array;
// };
typedef union {
  uint8_t uint8;
  int8_t int8;
  uint16_t uint16;
  int16_t int16;
  uint32_t uint32;
  int32_t int32;
  float float32;
  uint64_t uint64;
  int64_t int64;
  double float64;
  bool bool_;
  iree_string_view_t string;
  struct {
    gguf_metadata_value_type_t type;
    uint64_t len;
    // Arrays ignored for now - we just skip them.
    // gguf_metadata_value_t array[/*len*/];
  } array;
} gguf_metadata_value_t;

// NOTE: value.string/value.array has interior variable length data (ew).
// struct gguf_metadata_kv_t {
//   gguf_string_t key;
//   gguf_metadata_value_type_t value_type;
//   gguf_metadata_value_t value;
// };
typedef struct {
  iree_string_view_t key;
  gguf_metadata_value_type_t value_type;
  gguf_metadata_value_t value;
} gguf_metadata_kv_t;

// NOTE: metadata_kv has interior variable length data (ew).
// struct gguf_header_t {
//   uint32_t magic;
//   uint32_t version;
//   uint64_t tensor_count;
//   uint64_t metadata_kv_count;
//   gguf_metadata_kv_t metadata_kv[metadata_kv_count];
// };

// NOTE: storage has interior variable length data (ew).
// struct gguf_tensor_info_t {
//   gguf_string_t name;
//   uint32_t n_dimensions;
//   uint64_t dimensions[n_dimensions];
//   ggml_type_t type;
//   uint64_t offset;
// };
typedef struct {
  iree_string_view_t name;
  uint32_t n_dimensions;
  const uint64_t* dimensions;  // n_dimensions
  ggml_type_t type;
  uint64_t offset;
} gguf_tensor_info_t;

// struct gguf_file_t {
//   gguf_header_t header;
//   gguf_tensor_info_t tensor_infos[header.tensor_count];
//   uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
//   uint8_t tensor_data[];
// };

typedef struct iree_io_gguf_parser_t {
  // Handle of the file being parsed.
  iree_io_file_handle_t* file_handle;
  // Index being appended to during parsing.
  iree_io_parameter_index_t* index;
  // Default value GGUF_DEFAULT_ALIGNMENT or the general.alignment kv value.
  // The file tensor_data must be aligned to this value as must every tensor
  // contained within it.
  uint32_t alignment;
  // Offset of the tensor_data file field. A 0 value (file offset 0) indicates
  // the tensor_data offset has not been calculated yet. GGUF is nested variable
  // length structs and unfortunately is not possible to scan in a single pass.
  // Based off of the origin of the file.
  uint64_t tensor_data_offset;
  // Total available tensor data capacity. A 0 value indicates the
  // tensor_data_size (and the required tensor_data_offset) have not been
  // calculated yet.
  uint64_t tensor_data_size;
} iree_io_gguf_parser_t;

static inline uint64_t iree_align_uint64(uint64_t value, uint64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

static uint64_t iree_io_gguf_calculate_storage_size(
    const gguf_tensor_info_t* tensor_info) {
  uint64_t element_count = 1;
  for (uint32_t i = 0; i < tensor_info->n_dimensions; ++i) {
    element_count *= tensor_info->dimensions[i];
  }
  const ggml_type_traits_t type_traits = ggml_type_traits[tensor_info->type];
  return (element_count * type_traits.type_size) / type_traits.blck_size;
}

static iree_status_t iree_io_gguf_parse_value(iree_const_byte_span_t* contents,
                                              iree_host_size_t length,
                                              void* out_value) {
  if (contents->data_length < length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "file buffer underrun parsing %" PRIhsz " byte value", length);
  }
  memcpy(out_value, contents->data, length);
  contents->data += length;
  contents->data_length -= length;
  return iree_ok_status();
}
static iree_status_t iree_io_gguf_parse_uint32(iree_const_byte_span_t* contents,
                                               uint32_t* out_value) {
  return iree_io_gguf_parse_value(contents, sizeof(*out_value), out_value);
}
static iree_status_t iree_io_gguf_parse_uint64(iree_const_byte_span_t* contents,
                                               uint64_t* out_value) {
  return iree_io_gguf_parse_value(contents, sizeof(*out_value), out_value);
}

static iree_status_t iree_io_gguf_parse_array(iree_const_byte_span_t* contents,
                                              uint64_t element_count,
                                              uint64_t element_size,
                                              const uint8_t** out_base_ptr) {
  uint64_t total_length = element_count * element_size;
  if (total_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "attempting to load a 64-bit file on a 32-bit arch "
                            "(out of bounds array length)");
  }
  if (contents->data_length < (iree_host_size_t)total_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file buffer underrun parsing array");
  }
  *out_base_ptr = contents->data;
  contents->data += (iree_host_size_t)total_length;
  contents->data_length -= (iree_host_size_t)total_length;
  return iree_ok_status();
}

static iree_status_t iree_io_gguf_parse_string(iree_const_byte_span_t* contents,
                                               iree_string_view_t* out_value) {
  uint64_t length = 0;
  IREE_RETURN_IF_ERROR(iree_io_gguf_parse_uint64(contents, &length));
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "attempting to load a 64-bit file on a 32-bit arch "
                            "(out of bounds string length)");
  }
  out_value->size = (iree_host_size_t)length;
  return iree_io_gguf_parse_array(contents, length, sizeof(char),
                                  (const uint8_t**)&out_value->data);
}

static iree_status_t iree_io_gguf_skip_metadata_array(
    iree_const_byte_span_t* contents, gguf_metadata_value_type_t value_type,
    uint64_t count) {
  switch (value_type) {
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported metadata value type %u", value_type);
    case GGUF_METADATA_VALUE_TYPE_UINT8:
    case GGUF_METADATA_VALUE_TYPE_INT8:
    case GGUF_METADATA_VALUE_TYPE_UINT16:
    case GGUF_METADATA_VALUE_TYPE_INT16:
    case GGUF_METADATA_VALUE_TYPE_UINT32:
    case GGUF_METADATA_VALUE_TYPE_INT32:
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
    case GGUF_METADATA_VALUE_TYPE_BOOL:
    case GGUF_METADATA_VALUE_TYPE_UINT64:
    case GGUF_METADATA_VALUE_TYPE_INT64:
    case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
      const uint8_t* values = NULL;
      return iree_io_gguf_parse_array(
          contents, count, gguf_metadata_value_type_sizes[value_type], &values);
    }
    case GGUF_METADATA_VALUE_TYPE_STRING: {
      iree_string_view_t value = iree_string_view_empty();
      for (uint64_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(iree_io_gguf_parse_string(contents, &value));
      }
      return iree_ok_status();
    }
    case GGUF_METADATA_VALUE_TYPE_ARRAY:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "nested arrays not supported in gguf");
  }
}

static iree_status_t iree_io_gguf_parse_metadata_value(
    iree_const_byte_span_t* contents, gguf_metadata_value_type_t value_type,
    gguf_metadata_value_t* out_value) {
  switch (value_type) {
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported metadata value type %u", value_type);
    case GGUF_METADATA_VALUE_TYPE_UINT8:
    case GGUF_METADATA_VALUE_TYPE_INT8:
    case GGUF_METADATA_VALUE_TYPE_UINT16:
    case GGUF_METADATA_VALUE_TYPE_INT16:
    case GGUF_METADATA_VALUE_TYPE_UINT32:
    case GGUF_METADATA_VALUE_TYPE_INT32:
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
    case GGUF_METADATA_VALUE_TYPE_BOOL:
    case GGUF_METADATA_VALUE_TYPE_UINT64:
    case GGUF_METADATA_VALUE_TYPE_INT64:
    case GGUF_METADATA_VALUE_TYPE_FLOAT64:
      return iree_io_gguf_parse_value(
          contents, gguf_metadata_value_type_sizes[value_type], out_value);
    case GGUF_METADATA_VALUE_TYPE_STRING:
      return iree_io_gguf_parse_string(contents, &out_value->string);
    case GGUF_METADATA_VALUE_TYPE_ARRAY: {
      IREE_RETURN_IF_ERROR(
          iree_io_gguf_parse_uint32(contents, &out_value->array.type));
      IREE_RETURN_IF_ERROR(
          iree_io_gguf_parse_uint64(contents, &out_value->array.len));
      // We don't support arrays right now because they require allocation due
      // to the variable length nature of things. We do still have to calculate
      // the total size which is annoying due to the nested variable-length
      // strings.
      return iree_io_gguf_skip_metadata_array(contents, out_value->array.type,
                                              out_value->array.len);
    }
  }
}

typedef iree_status_t(IREE_API_PTR* iree_io_gguf_metadata_kv_callback_fn_t)(
    void* user_data, const gguf_metadata_kv_t* kv);
static iree_status_t iree_io_gguf_enumerate_metadata_kv(
    iree_const_byte_span_t* contents, uint64_t count,
    iree_io_gguf_metadata_kv_callback_fn_t callback, void* user_data) {
  for (uint64_t i = 0; i < count; ++i) {
    gguf_metadata_kv_t kv = {0};
    IREE_RETURN_IF_ERROR(iree_io_gguf_parse_string(contents, &kv.key));
    IREE_RETURN_IF_ERROR(iree_io_gguf_parse_uint32(contents, &kv.value_type));
    IREE_RETURN_IF_ERROR(
        iree_io_gguf_parse_metadata_value(contents, kv.value_type, &kv.value));
    IREE_RETURN_IF_ERROR(callback(user_data, &kv));
  }
  return iree_ok_status();
}

typedef iree_status_t(IREE_API_PTR* iree_io_gguf_tensor_info_callback_fn_t)(
    void* user_data, const gguf_tensor_info_t* tensor_info);
static iree_status_t iree_io_gguf_enumerate_tensor_info(
    iree_const_byte_span_t* contents, uint64_t count,
    iree_io_gguf_tensor_info_callback_fn_t callback, void* user_data) {
  for (uint64_t i = 0; i < count; ++i) {
    gguf_tensor_info_t tensor_info = {0};
    IREE_RETURN_IF_ERROR(
        iree_io_gguf_parse_string(contents, &tensor_info.name));
    IREE_RETURN_IF_ERROR(
        iree_io_gguf_parse_uint32(contents, &tensor_info.n_dimensions));
    IREE_RETURN_IF_ERROR(iree_io_gguf_parse_array(
        contents, tensor_info.n_dimensions, sizeof(tensor_info.dimensions[0]),
        (const uint8_t**)&tensor_info.dimensions));
    IREE_RETURN_IF_ERROR(
        iree_io_gguf_parse_uint32(contents, &tensor_info.type));
    IREE_RETURN_IF_ERROR(
        iree_io_gguf_parse_uint64(contents, &tensor_info.offset));
    if (callback) {
      IREE_RETURN_IF_ERROR(callback(user_data, &tensor_info));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_io_gguf_parse_metadata(void* user_data,
                                                 const gguf_metadata_kv_t* kv) {
  iree_io_gguf_parser_t* parser = (iree_io_gguf_parser_t*)user_data;
  if (iree_string_view_equal(kv->key, IREE_SV("general.alignment"))) {
    if (kv->value_type != GGUF_METADATA_VALUE_TYPE_UINT32) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "general.alignment metadata value must be uint32");
    }
    parser->alignment = kv->value.uint32;
  }
  return iree_ok_status();
}

static iree_status_t iree_io_gguf_append_tensor_info(
    void* user_data, const gguf_tensor_info_t* tensor_info) {
  iree_io_gguf_parser_t* parser = (iree_io_gguf_parser_t*)user_data;

  // Unfortunately (I've said that a lot here?) the total size of the tensor is
  // not stored and as such we need to calculate it based on the metadata we
  // have. If they just included the size we wouldn't even have to care about
  // data type or tensor dimensions and not need to handle the ever-growing list
  // of hard-coded format types.
  uint64_t storage_size = iree_io_gguf_calculate_storage_size(tensor_info);

  // Verify the range is within tensor data bounds.
  uint64_t begin = tensor_info->offset;
  uint64_t end = begin + storage_size;
  if (begin > end || end > parser->tensor_data_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry has data offsets outside of the "
                            "available data (begin=%" PRIu64 ", end=%" PRIu64
                            ", available=%" PRIu64 ")",
                            begin, end, parser->tensor_data_size);
  }

  // Add entry to the index.
  iree_io_parameter_index_entry_t entry = {
      .key = tensor_info->name,
      .metadata = iree_const_byte_span_empty(),
      .file_handle = parser->file_handle,
      .offset = parser->tensor_data_offset + begin,
      .length = storage_size,
  };
  return iree_io_parameter_index_add(parser->index, &entry);
}

IREE_API_EXPORT iree_status_t iree_io_parse_gguf_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_parameter_index_t* index) {
  // Read the header enough to check for file validity and version.
  // Unfortunately the format has a variable-length header (vs being
  // table-based) and that means we have to actually parse the header fully
  // (including all nested variable-length elements) in order to even know if
  // the whole header is present or where data lives. Yuck.
  iree_const_byte_span_t contents = file_contents;
  uint32_t magic = 0;
  IREE_RETURN_IF_ERROR(iree_io_gguf_parse_uint32(&contents, &magic));
  if (magic != GGUF_MAGIC) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "GGUF file magic missing or invalid %08X; expected %08X", magic,
        GGUF_MAGIC);
  }
  uint32_t version = 0;
  IREE_RETURN_IF_ERROR(iree_io_gguf_parse_uint32(&contents, &version));
  if (version != GGUF_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GGUF format version %u is unsupported; expected version %u", version,
        GGUF_VERSION);
  }
  uint64_t tensor_count = 0;
  IREE_RETURN_IF_ERROR(iree_io_gguf_parse_uint64(&contents, &tensor_count));
  uint64_t metadata_kv_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_io_gguf_parse_uint64(&contents, &metadata_kv_count));

  // If there are no tensors then no-op the parse. Probably not what the user
  // wanted but it's legal.
  if (tensor_count == 0) return iree_ok_status();

  iree_io_gguf_parser_t parser = {
      .file_handle = file_handle,
      .index = index,
      .alignment = GGUF_DEFAULT_ALIGNMENT,  // may be overridden
      .tensor_data_offset = 0,              // to be calculated
      .tensor_data_size = 0,                // to be calculated
  };

  // Scope data to the remainder of the file and enumerate all metadata pairs.
  // Upon return the contents will start immediately after the header and at
  // the start of tensor info.
  IREE_RETURN_IF_ERROR(iree_io_gguf_enumerate_metadata_kv(
      &contents, metadata_kv_count, iree_io_gguf_parse_metadata, &parser));

  // Scan forward through the tensor info to find where the tensor data base
  // offset is in the file. Unfortunately GGUF was designed without this offset
  // and because tensor info is variable length we cannot determine absolute
  // file offsets without doing two scans.
  iree_const_byte_span_t tensor_info_contents = contents;
  IREE_RETURN_IF_ERROR(iree_io_gguf_enumerate_tensor_info(
      &tensor_info_contents, tensor_count, NULL, &parser));

  // Calculate where the tensor data begins in the file. This respects the
  // default alignment or the general.alignment specified by the file.
  parser.tensor_data_offset = iree_align_uint64(
      (uint64_t)(tensor_info_contents.data - file_contents.data),
      parser.alignment);
  parser.tensor_data_size =
      file_contents.data_length - parser.tensor_data_offset;

  // Scan forward through the tensor info now that we know the tensor data
  // offset and add the tensor entries.
  IREE_RETURN_IF_ERROR(iree_io_gguf_enumerate_tensor_info(
      &contents, tensor_count, iree_io_gguf_append_tensor_info, &parser));

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parse_gguf_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Today we only support memory files.
  // TODO(benvanik): support iree_io_stream_t wrapping for parsing the index.
  if (iree_io_file_handle_type(file_handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-memory gguf files not yet supported");
  }
  iree_byte_span_t host_allocation =
      iree_io_file_handle_primitive(file_handle).value.host_allocation;

  iree_status_t status = iree_io_parse_gguf_index_from_memory(
      file_handle,
      iree_make_const_byte_span(host_allocation.data,
                                host_allocation.data_length),
      index);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
