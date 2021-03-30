// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_BASE_INTERNAL_DYNAMIC_LIBRARY_H_
#define IREE_BASE_INTERNAL_DYNAMIC_LIBRARY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Defines the behavior of the dynamic library loader.
enum iree_dynamic_library_flags_e {
  IREE_DYNAMIC_LIBRARY_FLAG_NONE = 0u,
};
typedef uint32_t iree_dynamic_library_flags_t;

// Dynamic library (aka shared object) cross-platform wrapper.
typedef struct iree_dynamic_library_s iree_dynamic_library_t;

// Loads a system library using both the system library load paths and the given
// file name. The path may may be absolute or relative.
//
// For process-wide search control the LD_LIBRARY_PATH (Linux) or PATH (Windows)
// is used in addition to the default search path rules of the platform.
iree_status_t iree_dynamic_library_load_from_file(
    const char* file_path, iree_dynamic_library_flags_t flags,
    iree_allocator_t allocator, iree_dynamic_library_t** out_library);

// Loads a system library using both the system library load paths and the given
// search path/alternative file names. The paths may may be absolute or
// relative.
//
// For process-wide search control the LD_LIBRARY_PATH (Linux) or PATH (Windows)
// is used in addition to the default search path rules of the platform.
iree_status_t iree_dynamic_library_load_from_files(
    iree_host_size_t search_path_count, const char* const* search_paths,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library);

// Opens a dynamic library from a range of bytes in memory.
// |identifier| will be used as the module name in debugging/profiling tools.
// |buffer| must remain live for the lifetime of the library.
iree_status_t iree_dynamic_library_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library);

// Retains the given |library| for the caller.
void iree_dynamic_library_retain(iree_dynamic_library_t* library);

// Releases the given |library| from the caller.
void iree_dynamic_library_release(iree_dynamic_library_t* library);

// Performs a symbol lookup in the dynamic library exports.
iree_status_t iree_dynamic_library_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn);

// Loads a debug database (PDB/DWARF/etc) from the given path providing debug
// symbols for this library and attaches it to the symbol store (if active).
iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path);

// Loads a debug database (PDB/DWARF/etc) from a range of bytes in memory and
// attaches it to the symbol store (if active). |buffer| must remain live for
// the lifetime of the library.
iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_DYNAMIC_LIBRARY_H_
