// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/libaqlprofile.h"

#include "aqlprofile-sdk/aql_profile_v2.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

// Keep vendor aqlprofile headers private to this translation unit. The public
// wrapper mirrors only the narrow ABI surface used by the HAL so callers do not
// inherit a large global vendor namespace.
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_aqlprofile_handle_t) ==
                              sizeof(aqlprofile_handle_t),
                          "aqlprofile handle layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_aqlprofile_version_t) ==
                              sizeof(aqlprofile_version_t),
                          "aqlprofile version layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_buffer_desc_flags_t) ==
        sizeof(aqlprofile_buffer_desc_flags_t),
    "aqlprofile buffer flags layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_att_parameter_t) ==
        sizeof(aqlprofile_att_parameter_t),
    "aqlprofile ATT parameter layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_att_profile_t) ==
        sizeof(aqlprofile_att_profile_t),
    "aqlprofile ATT profile layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_pmc_event_flags_t) ==
        sizeof(aqlprofile_pmc_event_flags_t),
    "aqlprofile PMC event flags layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_aqlprofile_pmc_event_t) ==
                              sizeof(aqlprofile_pmc_event_t),
                          "aqlprofile PMC event layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_aqlprofile_agent_info_v1_t) ==
                              sizeof(aqlprofile_agent_info_v1_t),
                          "aqlprofile agent info layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_agent_handle_t) ==
        sizeof(aqlprofile_agent_handle_t),
    "aqlprofile agent handle layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_pmc_profile_t) ==
        sizeof(aqlprofile_pmc_profile_t),
    "aqlprofile PMC profile layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_pmc_aql_packets_t) ==
        sizeof(aqlprofile_pmc_aql_packets_t),
    "aqlprofile PMC AQL packet bundle layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_att_control_aql_packets_t) ==
        sizeof(aqlprofile_att_control_aql_packets_t),
    "aqlprofile ATT AQL packet bundle layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_aqlprofile_att_code_object_data_t) ==
        sizeof(aqlprofile_att_codeobj_data_t),
    "aqlprofile ATT code object marker layout must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hsa_amd_aql_pm4_ib_packet_t) ==
        sizeof(hsa_ext_amd_aql_pm4_packet_t),
    "PM4-IB AQL packet layout must match the HSA vendor SDK");
IREE_AMDGPU_STATIC_ASSERT((uint32_t)IREE_HAL_AMDGPU_AQLPROFILE_BLOCK_NAME_SQ ==
                              (uint32_t)HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,
                          "SQ block id must match the HSA vendor SDK");
IREE_AMDGPU_STATIC_ASSERT(
    (uint32_t)IREE_HAL_AMDGPU_AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH ==
        (uint32_t)AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH,
    "ATT buffer-size-high parameter id must match the v2 SDK");
IREE_AMDGPU_STATIC_ASSERT(
    (uint32_t)IREE_HAL_AMDGPU_AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP ==
        (uint32_t)AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP,
    "ATT runtime-timestamp parameter id must match the v2 SDK");

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static const char* iree_hal_amdgpu_libaqlprofile_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "hsa-amd-aqlprofile64.dll",
#else
    // Versioned soname first: this is present in normal runtime packages. The
    // unversioned .so is usually only present in development installs.
    "libhsa-amd-aqlprofile64.so.1",
    "libhsa-amd-aqlprofile64.so",
#endif  // IREE_PLATFORM_WINDOWS
};

static iree_status_t iree_hal_amdgpu_libaqlprofile_load_symbols(
    iree_dynamic_library_t* library,
    iree_hal_amdgpu_libaqlprofile_t* out_libaqlprofile) {
#define IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(symbol)       \
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol( \
      library, #symbol, (void**)&out_libaqlprofile->symbol))

  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_get_version);
  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_register_agent_info);
  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_validate_pmc_event);
  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_pmc_create_packets);
  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_pmc_delete_packets);
  IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP(aqlprofile_pmc_iterate_data);

  out_libaqlprofile->aqlprofile_att_create_packets = (hsa_status_t(HSA_API*)(
      iree_hal_amdgpu_aqlprofile_handle_t*,
      iree_hal_amdgpu_aqlprofile_att_control_aql_packets_t*,
      iree_hal_amdgpu_aqlprofile_att_profile_t,
      iree_hal_amdgpu_aqlprofile_memory_alloc_callback_t,
      iree_hal_amdgpu_aqlprofile_memory_dealloc_callback_t,
      iree_hal_amdgpu_aqlprofile_memory_copy_callback_t, void*))
      iree_dynamic_library_try_lookup_symbol(library,
                                             "aqlprofile_att_create_packets");
  out_libaqlprofile->aqlprofile_att_delete_packets =
      (void(HSA_API*)(iree_hal_amdgpu_aqlprofile_handle_t))
          iree_dynamic_library_try_lookup_symbol(
              library, "aqlprofile_att_delete_packets");
  out_libaqlprofile->aqlprofile_att_iterate_data = (hsa_status_t(HSA_API*)(
      iree_hal_amdgpu_aqlprofile_handle_t,
      iree_hal_amdgpu_aqlprofile_att_data_callback_t, void*))
      iree_dynamic_library_try_lookup_symbol(library,
                                             "aqlprofile_att_iterate_data");
  out_libaqlprofile->aqlprofile_att_codeobj_marker = (hsa_status_t(HSA_API*)(
      iree_hsa_amd_aql_pm4_ib_packet_t*, iree_hal_amdgpu_aqlprofile_handle_t*,
      iree_hal_amdgpu_aqlprofile_att_code_object_data_t,
      iree_hal_amdgpu_aqlprofile_memory_alloc_callback_t,
      iree_hal_amdgpu_aqlprofile_memory_dealloc_callback_t, void*))
      iree_dynamic_library_try_lookup_symbol(library,
                                             "aqlprofile_att_codeobj_marker");

  out_libaqlprofile->hsa_ven_amd_aqlprofile_error_string =
      (hsa_status_t(HSA_API*)(const char**))
          iree_dynamic_library_try_lookup_symbol(
              library, "hsa_ven_amd_aqlprofile_error_string");

#undef IREE_HAL_AMDGPU_LIBAQLPROFILE_LOOKUP

  return iree_ok_status();
}

static bool iree_hal_amdgpu_libaqlprofile_try_load_library_from_file(
    const char* file_path, iree_string_builder_t* error_builder,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, file_path);
  *out_library = NULL;

  iree_status_t status = iree_dynamic_library_load_from_file(
      file_path, IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, out_library);
  if (!iree_status_is_ok(status)) {
    IREE_IGNORE_ERROR(iree_string_builder_append_format(
        error_builder, "\n  Tried: %s\n    ", file_path));
    IREE_IGNORE_ERROR(iree_string_builder_append_status(error_builder, status));
  }

  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static bool iree_hal_amdgpu_libaqlprofile_try_load_library_from_path(
    iree_string_view_t path_fragment, iree_string_builder_t* error_builder,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path_fragment.data, path_fragment.size);
  *out_library = NULL;

  iree_string_builder_t path_builder;
  iree_string_builder_initialize(host_allocator, &path_builder);

  if (iree_file_path_is_dynamic_library(path_fragment)) {
    IREE_IGNORE_ERROR(
        iree_string_builder_append_string(&path_builder, path_fragment));
    iree_hal_amdgpu_libaqlprofile_try_load_library_from_file(
        iree_string_builder_buffer(&path_builder), error_builder,
        host_allocator, out_library);
  } else {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_hal_amdgpu_libaqlprofile_names) &&
         !*out_library;
         ++i) {
      iree_string_builder_reset(&path_builder);
      IREE_IGNORE_ERROR(iree_string_builder_append_format(
          &path_builder, "%.*s/%s", (int)path_fragment.size, path_fragment.data,
          iree_hal_amdgpu_libaqlprofile_names[i]));
      path_builder.size = iree_file_path_canonicalize(
          (char*)iree_string_builder_buffer(&path_builder),
          iree_string_builder_size(&path_builder));
      iree_hal_amdgpu_libaqlprofile_try_load_library_from_file(
          iree_string_builder_buffer(&path_builder), error_builder,
          host_allocator, out_library);
    }
  }

  iree_string_builder_deinitialize(&path_builder);

  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static iree_status_t iree_hal_amdgpu_libaqlprofile_try_load_adjacent_to_libhsa(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_string_builder_t* error_builder, iree_allocator_t host_allocator,
    iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  iree_string_builder_t path_builder;
  iree_string_builder_initialize(host_allocator, &path_builder);
  iree_status_t status =
      iree_hal_amdgpu_libhsa_append_path_to_builder(libhsa, &path_builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t libhsa_path = iree_string_builder_view(&path_builder);
    iree_string_view_t libhsa_dirname = iree_file_path_dirname(libhsa_path);
    if (!iree_string_view_is_empty(libhsa_dirname)) {
      iree_hal_amdgpu_libaqlprofile_try_load_library_from_path(
          libhsa_dirname, error_builder, host_allocator, out_library);
    }
  }
  iree_string_builder_deinitialize(&path_builder);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_libaqlprofile_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_libaqlprofile_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_amdgpu_libaqlprofile_t* out_libaqlprofile) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_libaqlprofile);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_libaqlprofile, 0, sizeof(*out_libaqlprofile));

  iree_string_builder_t error_builder;
  iree_string_builder_initialize(host_allocator, &error_builder);

  iree_dynamic_library_t* library = NULL;
  for (iree_host_size_t i = 0; i < search_paths.count && !library; ++i) {
    iree_hal_amdgpu_libaqlprofile_try_load_library_from_path(
        search_paths.values[i], &error_builder, host_allocator, &library);
  }

  iree_string_view_t env_path =
      iree_make_cstring_view(getenv("IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH"));
  if (!library && !iree_string_view_is_empty(env_path)) {
    iree_hal_amdgpu_libaqlprofile_try_load_library_from_path(
        env_path, &error_builder, host_allocator, &library);
  }

  iree_status_t status = iree_ok_status();
  if (!library) {
    status = iree_hal_amdgpu_libaqlprofile_try_load_adjacent_to_libhsa(
        libhsa, &error_builder, host_allocator, &library);
  }
  if (!library) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_hal_amdgpu_libaqlprofile_names) && !library;
         ++i) {
      iree_hal_amdgpu_libaqlprofile_try_load_library_from_file(
          iree_hal_amdgpu_libaqlprofile_names[i], &error_builder,
          host_allocator, &library);
    }
  }

  if (iree_status_is_ok(status) && !library) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU aqlprofile library not found; hardware counter profiling "
        "requires libhsa-amd-aqlprofile64 to be installed next to HSA, on a "
        "system search path, or specified with "
        "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH: %.*s",
        (int)iree_string_builder_size(&error_builder),
        iree_string_builder_buffer(&error_builder));
  }
  iree_string_builder_deinitialize(&error_builder);

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_libaqlprofile_load_symbols(library, out_libaqlprofile);
  }
  if (iree_status_is_ok(status)) {
    out_libaqlprofile->library = library;
  } else {
    iree_dynamic_library_release(library);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_libaqlprofile_deinitialize(
    iree_hal_amdgpu_libaqlprofile_t* libaqlprofile) {
  IREE_ASSERT_ARGUMENT(libaqlprofile);
  iree_dynamic_library_release(libaqlprofile->library);
  memset(libaqlprofile, 0, sizeof(*libaqlprofile));
}

bool iree_hal_amdgpu_libaqlprofile_has_att_support(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile) {
  return libaqlprofile && libaqlprofile->aqlprofile_att_create_packets &&
         libaqlprofile->aqlprofile_att_delete_packets &&
         libaqlprofile->aqlprofile_att_iterate_data &&
         libaqlprofile->aqlprofile_att_codeobj_marker;
}

iree_status_t iree_status_from_aqlprofile_status(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile, const char* file,
    uint32_t line, hsa_status_t hsa_status, const char* symbol,
    const char* message) {
  if (hsa_status == HSA_STATUS_SUCCESS) return iree_ok_status();

  const char* error_string = NULL;
  if (libaqlprofile && libaqlprofile->hsa_ven_amd_aqlprofile_error_string) {
    hsa_status_t error_string_status =
        libaqlprofile->hsa_ven_amd_aqlprofile_error_string(&error_string);
    if (error_string_status != HSA_STATUS_SUCCESS) {
      error_string = NULL;
    }
  }
  if (!error_string) {
    error_string = "unknown aqlprofile error";
  }

  return iree_make_status_with_location(
      file, line, IREE_STATUS_INTERNAL,
      "%s failed with hsa_status=0x%08X (%s)%s%s", symbol, (uint32_t)hsa_status,
      error_string, message ? ": " : "", message ? message : "");
}
