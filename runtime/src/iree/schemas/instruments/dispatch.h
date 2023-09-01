// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SCHEMAS_INSTRUMENTS_DISPATCH_H_
#define IREE_SCHEMAS_INSTRUMENTS_DISPATCH_H_

#include <stdint.h>

//===----------------------------------------------------------------------===//
// iree_idbts_chunk_t
//===----------------------------------------------------------------------===//
// Represents a single chunk of data within a larger chunked transport stream.
// Each chunk has a type and length to allow quick scans for chunks of interest.
// Chunks must always start at 16-byte alignment boundaries.
//
// TODO(benvanik): clean up and move to a dedicated idbts.h.
// This is temporarily here while iterating on the transport stream.

// Chunk magic identifier.
// "IREE Instrumentation Database v0"
// "IDB0" = 0x49 0x44 0x42 0x30
typedef uint32_t iree_idbts_chunk_magic_t;
#define IREE_IDBTS_CHUNK_MAGIC 0x42444930u

// Chunk type.
enum iree_idbts_chunk_type_e {
  // NOTE: these will change in the real IDB spec.
  IREE_IDBTS_CHUNK_TYPE_DISPATCH_METADATA = 0x0000u,
  IREE_IDBTS_CHUNK_TYPE_DISPATCH_RINGBUFFER = 0x0001u,
};
typedef uint16_t iree_idbts_chunk_type_t;

// IDB chunk format version.
// Instruments and other embedded chunks may version themselves independently to
// prevent entire files from being invalidated on compiler bumps.
typedef uint16_t iree_idbts_chunk_version_t;

// Header at the prefix of each chunk in the file.
// Always aligned to 16-bytes in the file such that the trailing chunk contents
// are 16-byte aligned.
typedef struct {
  // Magic header bytes; must be IREE_IDBTS_CHUNK_MAGIC.
  iree_idbts_chunk_magic_t magic;
  // Type of the chunk used to interpret the payload.
  iree_idbts_chunk_type_t type;
  // Type-specific version identifier. Usually 0.
  iree_idbts_chunk_version_t version;
  // Total byte length of the chunk content excluding this header.
  uint64_t content_length;
} iree_idbts_chunk_header_t;
static_assert(sizeof(iree_idbts_chunk_header_t) % 16 == 0,
              "chunk header must be 16-byte aligned");

//===----------------------------------------------------------------------===//
// Dispatch instrumentation ringbuffer transport stream
//===----------------------------------------------------------------------===//

// Total padding added to the base power-of-two ringbuffer size.
// The last 8 bytes are the monotonically increasing ringbuffer write head.
// Up to the (padding-8) bytes are available for the ringbuffer to spill past
// its end.
#define IREE_INSTRUMENT_DISPATCH_PADDING 4096

typedef enum iree_instrument_dispatch_type_e {
  IREE_INSTRUMENT_DISPATCH_TYPE_WORKGROUP = 0b00000000,
  IREE_INSTRUMENT_DISPATCH_TYPE_PRINT = 0b00000001,
  IREE_INSTRUMENT_DISPATCH_TYPE_VALUE = 0b00000010,
  IREE_INSTRUMENT_DISPATCH_TYPE_RESERVED_0 = 0b00000011,  // free for use
  IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_LOAD = 0b00000100,
  IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_STORE = 0b00000101,
} iree_instrument_dispatch_type_t;

typedef struct iree_instrument_dispatch_header_t {
  uint32_t tag : 8;
  uint32_t unknown : 24;
} iree_instrument_dispatch_header_t;

typedef struct iree_instrument_dispatch_workgroup_t {
  uint32_t tag : 8;  // IREE_INSTRUMENT_DISPATCH_TYPE_WORKGROUP
  uint32_t dispatch_id : 24;
  uint32_t workgroup_id_x;
  uint32_t workgroup_id_y;
  uint32_t workgroup_id_z;
  uint32_t workgroup_count_x;
  uint32_t workgroup_count_y;
  uint32_t workgroup_count_z;
  uint32_t processor_id;
} iree_instrument_dispatch_workgroup_t;

typedef struct iree_instrument_dispatch_print_t {
  uint64_t tag : 8;  // IREE_INSTRUMENT_DISPATCH_TYPE_PRINT
  uint64_t length : 16;
  uint64_t workgroup_offset : 40;
  uint8_t data[];  // length, padded to ensure 16b struct, no NUL terminator
} iree_instrument_dispatch_print_t;

typedef struct iree_instrument_dispatch_memory_op_t {
  uint64_t
      tag : 8;  // IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_LOAD / _MEMORY_STORE
  uint64_t length : 16;
  uint64_t workgroup_offset : 40;
  uint64_t address;
} iree_instrument_dispatch_memory_op_t;

enum iree_instrument_dispatch_value_type_e {
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_8 = 0,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_8,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_16,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_16,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_32,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_32,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_64,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_POINTER,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_16,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_32,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_64,
  IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_BFLOAT_16,
};
typedef uint64_t iree_instrument_dispatch_value_type_t;

typedef struct iree_instrument_dispatch_value_t {
  uint64_t tag : 8;   // IREE_INSTRUMENT_DISPATCH_TYPE_VALUE
  uint64_t type : 8;  // iree_instrument_dispatch_value_type_t
  uint64_t ordinal : 8;
  uint64_t workgroup_offset : 40;
  uint64_t bits;
} iree_instrument_dispatch_value_t;

#endif  // IREE_SCHEMAS_INSTRUMENTS_DISPATCH_H_
