// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_ELF_FORMAT_H_
#define IREE_HAL_UTILS_ELF_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true if |elf_data| begins with the ELF magic bytes.
//
// If |elf_data.data_length| is zero, |elf_data.data| is still inspected. This
// matches HAL executable format inference, where a zero data length means the
// total executable size is not yet known.
bool iree_hal_elf_data_starts_with_magic(iree_const_byte_span_t elf_data);

// Calculates the byte size of a 32-bit or 64-bit ELF file by inspecting its
// header and section table extents.
//
// If |elf_data.data_length| is zero, the function assumes the header and
// section table bytes it must inspect are accessible through |elf_data.data|.
// Callers use this mode only during executable format inference, before the HAL
// has established a bounded span for the whole executable.
iree_status_t iree_hal_elf_calculate_size(iree_const_byte_span_t elf_data,
                                          iree_host_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_ELF_FORMAT_H_
