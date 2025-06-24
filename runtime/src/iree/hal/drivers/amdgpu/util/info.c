// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/info.h"

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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major),
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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa), HSA_SYSTEM_INFO_ENDIANNESS,
                               &endianness),
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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_SYSTEM_INFO_MACHINE_MODEL, &machine_model),
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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED,
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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT,
                               &svm_accessible_by_default),
      "querying HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT");
  out_info->svm_accessible_by_default = svm_accessible_by_default ? 1 : 0;

  bool dmabuf_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED,
                               &dmabuf_supported),
      "querying HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED");
  out_info->dmabuf_supported = dmabuf_supported ? 1 : 0;

  bool virtual_mem_api_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
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
      iree_hsa_system_get_info(IREE_LIBHSA(libhsa),
                               HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                               &out_info->timestamp_frequency),
      "querying HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY");

  uint16_t amd_loader_minor = 0;
  bool amd_loader_supported = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_system_major_extension_supported(
          IREE_LIBHSA(libhsa), HSA_EXTENSION_AMD_LOADER, 1, &amd_loader_minor,
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
