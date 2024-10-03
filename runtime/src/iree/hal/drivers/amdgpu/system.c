// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_info_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_system_info_query(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_system_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_info, 0, sizeof(*out_info));

  uint16_t version_major = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_VERSION_MAJOR,
                               &version_major),
      "querying HSA_SYSTEM_INFO_VERSION_MAJOR");
  if (version_major != 1) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "only HSA 1.x is supported "
                             "(HSA_SYSTEM_INFO_VERSION_MAJOR == 1, have %u)",
                             version_major));
  }

  hsa_endianness_t endianness = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_ENDIANNESS, &endianness),
      "querying HSA_SYSTEM_INFO_ENDIANNESS");
  if (endianness != HSA_ENDIANNESS_LITTLE) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only little-endian systems are supported "
                "(HSA_SYSTEM_INFO_ENDIANNESS == HSA_ENDIANNESS_LITTLE)"));
  }

  hsa_machine_model_t machine_model = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_MACHINE_MODEL,
                               &machine_model),
      "querying HSA_SYSTEM_INFO_MACHINE_MODEL");
  if (machine_model != HSA_MACHINE_MODEL_LARGE) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only 64-bit systems are supported "
                "(HSA_SYSTEM_INFO_MACHINE_MODEL == HSA_MACHINE_MODEL_LARGE)"));
  }

  bool svm_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED,
                               &svm_supported),
      "querying HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED");
  if (!svm_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INCOMPATIBLE,
                             "only systems with SVM are supported "
                             "(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED == true)"));
  }

  bool svm_accessible_by_default = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa,
                               HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT,
                               &svm_accessible_by_default),
      "querying HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT");
  out_info->svm_accessible_by_default = svm_accessible_by_default ? 1 : 0;

  bool dmabuf_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED,
                               &dmabuf_supported),
      "querying HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED");
  out_info->dmabuf_supported = dmabuf_supported ? 1 : 0;

  bool virtual_mem_api_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa,
                               HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED,
                               &virtual_mem_api_supported),
      "querying HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED");
  if (!virtual_mem_api_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only systems with the virtual memory API are supported "
                "(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED == true)"));
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(libhsa, HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                               &out_info->timestamp_frequency),
      "querying HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY");

  uint16_t amd_loader_minor = 0;
  bool amd_loader_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_major_extension_supported(
          libhsa, HSA_EXTENSION_AMD_LOADER, 1, &amd_loader_minor,
          &amd_loader_supported),
      "querying HSA_EXTENSION_AMD_LOADER version");
  if (!amd_loader_supported) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_INCOMPATIBLE,
                "only systems with the AMD loader extension are supported "
                "(HSA_EXTENSION_AMD_LOADER v1.xx)"));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_system_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_t* out_system) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_system);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query and validate the system information.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_system_info_query(libhsa, &out_system->info));

  // Copy the libhsa symbol table and retain HSA for the lifetime of the system.
  // The caller may destroy the provided libhsa after this call returns.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_libhsa_copy(libhsa, &out_system->libhsa));

  // Copy the topology - today it's a plain-old-data struct and we can just
  // memcpy it. This is an implementation detail, though, and in the future if
  // it allocates anything we'll need to make sure this retains the allocations
  // or does a deep copy.
  memcpy(&out_system->topology, topology, sizeof(out_system->topology));

  // Initialize the device library, which will load the builtin executable and
  // fail if we don't have a supported arch.
  iree_status_t status = iree_hal_amdgpu_device_library_initialize(
      libhsa, topology, host_allocator, &out_system->device_library);

  // DO NOT SUBMIT pools?

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_system_deinitialize(out_system);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_system_deinitialize(iree_hal_amdgpu_system_t* system) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_device_library_deinitialize(&system->device_library);

  iree_hal_amdgpu_libhsa_deinitialize(&system->libhsa);

  IREE_TRACE_ZONE_END(z0);
}
