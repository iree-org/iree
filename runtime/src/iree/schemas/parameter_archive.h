// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SCHEMAS_PARAMETER_ARCHIVE_H_
#define IREE_SCHEMAS_PARAMETER_ARCHIVE_H_

#include <stdint.h>

//===----------------------------------------------------------------------===//
// IREE Parameter Archive
//===----------------------------------------------------------------------===//
//
// Each parameter archive stores zero or more parameters that are referenced by
// a string name. A single file may contain multiple archives linked together.
// Archives are designed for optimized concatenation (extend a file and link
// each archive header), appending (link new archive header), erasing (change
// entry to skip), and replacing (existing skip and then a new archive header).
//
// During mutation the archives may get spread out over the file and hurt
// startup performance. Repacking archive files with `iree-convert-parameters`
// will combine all archive headers at the head of the file to avoid needing to
// scan large files to index their contents.
//
// To enable concatenation all offset fields in an archive header are relative
// to the offset of the header added to the header-relative offset to the
// appropriate segment. Strings, paths, and metadata blobs are stored in a
// metadata segment that has no alignment specified. Parameters contents are
// stored separately in a data storage segment that meets the alignment
// requirements specified by each parameter.
//
// For aiding testing/benchmarking workflows involving large parameters the
// archive can contain entries backed by a splatted value instead of any real
// data. This allows for a program that references parameters to be compiled and
// run against a "fake" set of splatted parameters instead of needing the full
// original file on each machine running the tests/benchmarks.
// Use `iree-convert-parameters` with the `--strip` flag to strip all parameter
// values or `--splat=key` to strip selected parameters.

#if defined(_MSC_VER)
#define IREE_IO_PACKED_BEGIN __pragma(pack(push, 1))
#define IREE_IO_PACKED_END __pragma(pack(pop))
#else
#define IREE_IO_PACKED_BEGIN _Pragma("pack(push, 1)")
#define IREE_IO_PACKED_END _Pragma("pack(pop)")
#endif  // _MSC_VER

IREE_IO_PACKED_BEGIN

typedef uint64_t iree_io_physical_offset_t;
typedef uint64_t iree_io_physical_size_t;

// Preferred alignment for the file size containing an archive.
// This is the (common) small page size on most platforms and required in order
// to ensure fast-path memory mapping. Large pages and other configuration may
// require a larger alignment but we leave that to those special cases to
// handle.
#define IREE_IO_PARAMETER_ARCHIVE_DEFAULT_FILE_ALIGNMENT 4096

// Alignment of each entry in the archive entry table.
// Padding between entries in the table is zero filled.
#define IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT 16

// Alignment of each entry in the archive entry table.
// Padding between entries in the table is zero filled.
#define IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT 16

// Default alignment used for entry data storage.
#define IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT 64

// Parameter archive header magic identifier.
// "IREE Parameter Archive"
// "IRPA" = 0x49 0x52 0x50 0x41
typedef uint32_t iree_io_parameter_archive_magic_t;
#define IREE_IO_PARAMETER_ARCHIVE_MAGIC 0x41505249u

// Bitmask of flags denoting archive traits.
typedef uint64_t iree_io_parameter_archive_flags_t;

// A range of bytes relative to an offset as defined in the header.
typedef struct iree_io_parameter_archive_range_t {
  iree_io_physical_offset_t offset;
  iree_io_physical_size_t length;
} iree_io_parameter_archive_range_t;
// A range of bytes relative to the the header metadata_segment_offset.
typedef iree_io_parameter_archive_range_t
    iree_io_parameter_archive_metadata_ref_t;
// A range of bytes relative to the the header storage_segment_offset.
typedef iree_io_parameter_archive_range_t
    iree_io_parameter_archive_storage_ref_t;

// Header denoting the start of the of the archive in the enclosing file.
// Needs not start at file offset zero but must be aligned to a multiple of
// IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT. A single file may contain
// multiple archives by using the `next_header_offset` field to build a linked
// list of archive headers. All offsets in the header are relative to the header
// offset so a global offset is not required even with multiple archives.
//
// NOTE: the following fields must always exist in all header versions as
// parsers must be able to verify a file is an archive and check its version
// to emit nice errors. Parsers are allowed to skip headers they can't load so
// we also want to ensure the header size and any linked headers are available
// regardless of version.
typedef struct iree_io_parameter_archive_header_prefix_t {
  // Magic header bytes; must be IREE_IO_PARAMETER_ARCHIVE_MAGIC.
  iree_io_parameter_archive_magic_t magic;

  // Major version identifier; major versions are incompatible.
  uint16_t version_major;
  // Minor version identifier; new minor versions with the same major are
  // forward-compatible but may not be backwards compatible.
  uint16_t version_minor;

  // Total header size, including the 4 magic bytes.
  iree_io_physical_size_t header_size;

  // Optional (if 0) offset from the base of the header to another
  // iree_io_parameter_archive_header_t in the file. This allows files to be
  // concatenated or split without needing to re-encode any headers.
  iree_io_physical_offset_t next_header_offset;

  // Reserved for future use.
  iree_io_parameter_archive_flags_t flags;
} iree_io_parameter_archive_header_prefix_t;

// Archive header for version_major 0.
typedef struct iree_io_parameter_archive_header_v0_t {
  // Common header prefix.
  iree_io_parameter_archive_header_prefix_t prefix;

  // Total number of entries in the entry table.
  iree_io_physical_size_t entry_count;

  // Offset from the base of the header to where the entry table resides.
  // May overlap with other segments. Each entry in the table is aligned to
  // IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT.
  iree_io_parameter_archive_range_t entry_segment;

  // Offset from the base of the header to where the metadata segment resides.
  // May overlap with other segments.
  iree_io_parameter_archive_range_t metadata_segment;

  // Offset from the base of the header to where embedded data storage resides.
  // May overlap with other segments.
  iree_io_parameter_archive_range_t storage_segment;
} iree_io_parameter_archive_header_v0_t;

enum iree_io_parameter_archive_entry_type_e {
  // Entry is ignored during processing as if it didn't exist in the table.
  // Tools can erase entries by changing their type to this value.
  IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SKIP = 0,
  // Entry represents a repeating pattern splatted into memory.
  // See iree_io_parameter_archive_splat_entry_t.
  IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT = 1,
  // Entry represents data embedded in the archive.
  // See iree_io_parameter_archive_data_entry_t.
  IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA = 2,
  // Entry represents data stored in an external file.
  // See iree_io_parameter_archive_external_entry_t.
  IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_EXTERNAL = 3,
};
// Defines the type of an entry in the archive entry table.
typedef uint32_t iree_io_parameter_archive_entry_type_t;

// Entry type-specific flag bits.
typedef uint64_t iree_io_parameter_archive_entry_flags_t;

// Header shared across all entry types.
// The total size of the header and the entry are included to allow for
// traversing archives where not all entry types are known by the reader.
// Each entry must be aligned to IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT.
// Padding between entries in the table are zero-filled.
//
// Entry names are arbitrary strings (could be paths, nested identifiers, etc).
// Entry metadata is not defined by this format.
typedef struct iree_io_parameter_archive_entry_header_t {
  // Total size of the entry in the table, including this header but excluding
  // any trailing padding.
  iree_io_physical_size_t entry_size;
  // Type of the entry indicating the outer structure containing this header.
  iree_io_parameter_archive_entry_type_t type;
  // For use by each entry type.
  iree_io_parameter_archive_entry_flags_t flags;
  // Reference to a non-NUL-terminated entry name in the metadata segment.
  iree_io_parameter_archive_metadata_ref_t name;
  // Reference to a metadata blob in the metadata segment.
  // If the metadata is a string it need not have a NUL terminator.
  // A zero length indicates no metadata is present.
  iree_io_parameter_archive_metadata_ref_t metadata;
  // Minimum alignment required when stored in the parent file.
  // Implementations may require the absolute offset in the file to satisfy this
  // alignment. Zero if alignment is unspecified.
  iree_io_physical_size_t minimum_alignment;
} iree_io_parameter_archive_entry_header_t;

// An entry that contains a repeating pattern in virtual memory.
// The pattern length must evenly divide the entry length and be a power of
// two value of 1, 2, 4, 8, or 16 to allow for up to complex128 (complex<f64>).
typedef struct iree_io_parameter_archive_splat_entry_t {
  // Entry header with type IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT.
  iree_io_parameter_archive_entry_header_t header;
  // Total length of the splat parameter in bytes.
  iree_io_physical_size_t length;
  // Little-endian pattern used to fill the range in memory.
  // Examples:
  //         0xAA: pattern_length=1 pattern=[0xAA ...]
  //       0xBBAA: pattern_length=2 pattern=[0xAA 0xBB ...]
  //   0xDDCCBBAA: pattern_length=4 pattern=[0xAA 0xBB 0xCC 0xDD ...]
  uint8_t pattern[16];
  // Length of the pattern in bytes defining how many bytes of the pattern
  // field are valid from index 0.
  uint8_t pattern_length;
} iree_io_parameter_archive_splat_entry_t;

// An entry referencing a span of data in the archive data storage segment.
// Multiple entries with distinct virtual ranges may reference the same or
// overlapping physical ranges.
typedef struct iree_io_parameter_archive_data_entry_t {
  // Entry header with type IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA.
  iree_io_parameter_archive_entry_header_t header;
  // Relative offset of the data in the data storage segment.
  // Total length of the data in the data storage segment. If less than the
  // virtual length of the entry then it will be padded with zeros when loaded.
  iree_io_parameter_archive_storage_ref_t storage;
} iree_io_parameter_archive_data_entry_t;

// An entry referencing data in an external file.
typedef struct iree_io_parameter_archive_external_entry_t {
  // Entry header with type IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_EXTERNAL.
  iree_io_parameter_archive_entry_header_t header;
  // Reference to a non-NUL-terminated file path in the metadata segment.
  iree_io_parameter_archive_metadata_ref_t path;
  // Absolute offset and length of the data in the external file.
  iree_io_parameter_archive_range_t range;
} iree_io_parameter_archive_external_entry_t;

IREE_IO_PACKED_END

#endif  // IREE_SCHEMAS_PARAMETER_ARCHIVE_H_
