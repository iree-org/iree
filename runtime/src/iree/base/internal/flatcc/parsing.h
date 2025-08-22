// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_FLATCC_PARSING_H_
#define IREE_BASE_INTERNAL_FLATCC_PARSING_H_

#include <stdint.h>

//===----------------------------------------------------------------------===//
// flatcc include order fixes
//===----------------------------------------------------------------------===//
//
// This header merely wraps the flatcc headers that are generally useful to
// include in various places that may not know the specific messages they are
// working with.
//
// If using flatcc prefer to include this file over any hard-to-handle flatcc
// file such as flatbuffers_common_reader.h or flatbuffers_common_builder.h.
//
// NOTE: order matters for these includes so stop clang from messing with it:
// clang-format off

#include "flatcc/reflection/flatbuffers_common_reader.h"  // IWYU pragma: export
#include "iree/base/internal/flatcc/dummy_reader.h" // IWYU pragma: export

#include "flatcc/flatcc_verifier.h" // IWYU pragma: export
#include "iree/base/internal/flatcc/dummy_verifier.h" // IWYU pragma: export

// clang-format on

//===----------------------------------------------------------------------===//
// iree_flatbuffer_header_t
//===----------------------------------------------------------------------===//

// A header preceding flatbuffer data in a file.
// This is added by IREE tooling to allow for coarse versioning (in case we
// totally change the file format), flatbuffer sizing, and ensured alignment.
typedef struct iree_flatbuffer_file_header_t {
  // 4 byte magic number used for file identification.
  uint32_t magic;
  // Version of the content payload.
  uint32_t version;
  // Total size, in bytes, of the content following the header.
  // For flatbuffers this may include the flatbuffer itself plus any trailing
  // data referenced by it.
  uint64_t content_size;
  uint64_t reserved[6];
} iree_flatbuffer_file_header_t;
static_assert(sizeof(iree_flatbuffer_file_header_t) == 64,
              "must be 64-byte padded");

#endif  // IREE_BASE_INTERNAL_FLATCC_PARSING_H_
