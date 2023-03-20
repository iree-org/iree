// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ARCHIVE_H_
#define IREE_VM_BYTECODE_ARCHIVE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Alignment applied to each segment of the archive.
// All embedded file contents (FlatBuffers, rodata, etc) are aligned to this
// boundary.
#define IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT 64

// Parses the module archive header in |archive_contents|.
// The subrange containing the FlatBuffer data is returned as well as the
// offset where external rodata begins. Note that archives may have
// non-contiguous layouts!
IREE_API_EXPORT iree_status_t iree_vm_bytecode_archive_parse_header(
    iree_const_byte_span_t archive_contents,
    iree_const_byte_span_t* out_flatbuffer_contents,
    iree_host_size_t* out_rodata_offset);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ARCHIVE_H_
