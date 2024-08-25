// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HSA symbols
//===----------------------------------------------------------------------===//

#include <stdint.h>

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_init)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_shut_down)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_create, hsa_agent_t, uint32_t,
                               hsa_queue_type32_t,
                               void (*)(hsa_status_t, hsa_queue_t *, void *),
                               void *, uint32_t, uint32_t, hsa_queue_t **)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_wait_scacquire, hsa_signal_t,
                               hsa_signal_condition_t, hsa_signal_value_t,
                               uint64_t, hsa_wait_state_t)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_get_info, hsa_agent_t,
                               hsa_agent_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_iterate_agents,
                               hsa_status_t (*)(hsa_agent_t, void *), void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_load_write_index_relaxed,
                               const hsa_queue_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_create, hsa_signal_value_t, uint32_t,
                               const hsa_agent_t *, hsa_signal_t *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_store_write_index_release,
                               const hsa_queue_t *, uint64_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_add_write_index_relaxed,
                               const hsa_queue_t *, uint64_t)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_store_screlease, hsa_signal_t,
                               hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_store_relaxed, hsa_signal_t,
                               hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_add_screlease, hsa_signal_t,
                               hsa_signal_value_t)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_wait_acquire, hsa_signal_t,
                               hsa_signal_condition_t, hsa_signal_value_t,
                               uint64_t, hsa_wait_state_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_destroy, hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_get_symbol_by_name,
                               hsa_executable_t, const char *,
                               const hsa_agent_t *, hsa_executable_symbol_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_symbol_get_info,
                               hsa_executable_symbol_t,
                               hsa_executable_symbol_info_t, void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_ext_image_create, hsa_agent_t,
                               const hsa_ext_image_descriptor_t *, const void *,
                               hsa_access_permission_t, hsa_ext_image_t *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_create_alt, hsa_profile_t,
                               hsa_default_float_rounding_mode_t, const char *,
                               hsa_executable_t *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_load_agent_code_object,
                               hsa_executable_t, hsa_agent_t,
                               hsa_code_object_reader_t, const char *,
                               hsa_loaded_code_object_t *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_freeze, hsa_executable_t,
                               const char *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_destroy, hsa_executable_t)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_code_object_reader_create_from_memory,
                               const void *, size_t, hsa_code_object_reader_t *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_iterate_regions, hsa_agent_t,
                               hsa_status_t (*)(hsa_region_t, void *), void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_agent_iterate_memory_pools, hsa_agent_t,
                               hsa_status_t (*)(hsa_amd_memory_pool_t, void *),
                               void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_status_string, hsa_status_t, const char **)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_get_info,
                               hsa_amd_memory_pool_t,
                               hsa_amd_memory_pool_info_t, void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_region_get_info, hsa_region_t,
                               hsa_region_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_allocate, hsa_region_t, size_t,
                               void **)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_allocate,
                               hsa_amd_memory_pool_t, size_t, uint32_t, void **)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_free, void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_async_copy, void *, hsa_agent_t,
                               const void *, hsa_agent_t, size_t, uint32_t,
                               const hsa_signal_t *, hsa_signal_t)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_signal_async_handler, hsa_signal_t,
                               hsa_signal_condition_t, hsa_signal_value_t,
                               hsa_amd_signal_handler, void *)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_copy, void *, const void *, size_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_lock_to_pool, void *, size_t,
                               hsa_agent_t *, int, hsa_amd_memory_pool_t,
                               uint32_t, void **)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_fill, void *, uint32_t, size_t);
