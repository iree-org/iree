// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_FORMAT_H_
#define IREE_HAL_LOCAL_EXECUTABLE_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Infers the format and size of an ELF or FatELF executable.
// Returns format strings like "embedded-elf-x86_64" or "fatelf".
//
// |executable_data| contains the executable binary data. If data_length is 0
// the function will assume at least 4 bytes are readable for magic detection.
// This is UNSAFE but required for compatibility with APIs that don't provide
// size information.
//
// |executable_format_capacity| is the buffer size for the format string.
// |executable_format| receives the NUL-terminated format string.
//
// |out_inferred_size| receives the inferred size of the executable in bytes.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the format string buffer is too small.
// Returns IREE_STATUS_INCOMPATIBLE if the format is not ELF or FatELF.
IREE_API_EXPORT iree_status_t iree_hal_executable_infer_elf_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Infers the format and size of a PE/DLL executable.
// Detects Windows PE/DLL binaries and returns format strings like
// "system-dll-x86_64".
//
// |executable_data| contains the executable binary data. If data_length is 0,
// the function will assume at least 2 bytes are readable for magic detection.
// This is UNSAFE but required for compatibility with APIs that don't provide
// size information.
//
// |executable_format_capacity| is the buffer size for the format string.
// |executable_format| receives the NUL-terminated format string.
//
// |out_inferred_size| receives the inferred size of the executable in bytes.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the format string buffer is too small.
// Returns IREE_STATUS_INCOMPATIBLE if the format is not PE/DLL.
IREE_API_EXPORT iree_status_t iree_hal_executable_infer_dll_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Infers the format and size of a Mach-O executable.
// Detects Mach-O binaries and returns format strings like
// "system-dylib-x86_64".
//
// |executable_data| contains the executable binary data. If data_length is 0,
// the function will assume at least 4 bytes are readable for magic detection.
// This is UNSAFE but required for compatibility with APIs that don't provide
// size information.
//
// |executable_format_capacity| is the buffer size for the format string.
// |executable_format| receives the NUL-terminated format string.
//
// |out_inferred_size| receives the inferred size of the executable in bytes.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the format string buffer is too small.
// Returns IREE_STATUS_INCOMPATIBLE if the format is not Mach-O.
IREE_API_EXPORT iree_status_t iree_hal_executable_infer_macho_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Infers the format and size of a WebAssembly module.
// Detects WebAssembly modules and returns "wasm_32" for standard WASM or
// "wasm_64" for modules using memory64 (64-bit memory addressing).
//
// |executable_data| contains the executable binary data. If data_length is 0,
// the function will assume at least 4 bytes are readable for magic detection.
// This is UNSAFE but required for compatibility with APIs that don't provide
// size information.
//
// |executable_format_capacity| is the buffer size for the format string.
// |executable_format| receives the NUL-terminated format string.
//
// |out_inferred_size| receives the inferred size of the executable in bytes.
// Size inference for WASM may return 0 if not fully implemented.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the format string buffer is too small.
// Returns IREE_STATUS_INCOMPATIBLE if the format is not WebAssembly.
IREE_API_EXPORT iree_status_t iree_hal_executable_infer_wasm_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Infers the format and size of a system-native executable.
//
// Detects the native format for the current platform:
// - Windows: PE/DLL format ("system-dll-x86_64")
// - macOS: Mach-O format ("system-dylib-x86_64")
// - Emscripten: WebAssembly format ("wasm_32" or "wasm_64")
// - Linux/Unix: ELF/FatELF format ("system-elf-x86_64")
//
// |executable_data| contains the executable binary data. If data_length is 0,
// the function will assume at least 4 bytes are readable for magic detection.
// This is UNSAFE but required for compatibility with APIs that don't provide
// size information.
//
// |executable_format_capacity| is the buffer size for the format string.
// |executable_format| receives the NUL-terminated format string.
//
// |out_inferred_size| receives the inferred size of the executable in bytes.
// A size of 0 may be returned if size calculation is not supported on the
// current platform for the detected format.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the format string buffer is too small.
// Returns IREE_STATUS_INCOMPATIBLE if the format cannot be detected.
IREE_API_EXPORT iree_status_t iree_hal_executable_infer_system_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_FORMAT_H_
