// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HSA Core Runtime symbols
//===----------------------------------------------------------------------===//

// Initialization and shutdown
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_init)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_shut_down)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_status_string, hsa_status_t, const char**)

// System info
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_system_get_info, hsa_system_info_t, void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_system_get_major_extension_table,
                               uint16_t, uint16_t, size_t, void*)

// Agent (device) management
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_iterate_agents,
                               hsa_status_t (*)(hsa_agent_t, void*), void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_get_info, hsa_agent_t,
                               hsa_agent_info_t, void*)

// Queue management
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_create, hsa_agent_t, uint32_t,
                               hsa_queue_type32_t,
                               void (*)(hsa_status_t, hsa_queue_t*, void*),
                               void*, uint32_t, uint32_t, hsa_queue_t**)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_destroy, hsa_queue_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_load_read_index_relaxed,
                               const hsa_queue_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_load_write_index_relaxed,
                               const hsa_queue_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_store_write_index_relaxed,
                               const hsa_queue_t*, uint64_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_add_write_index_relaxed,
                               const hsa_queue_t*, uint64_t)

// Signal management
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_create, hsa_signal_value_t, uint32_t,
                               const hsa_agent_t*, hsa_signal_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_destroy, hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_load_relaxed, hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_load_scacquire, hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_store_relaxed, hsa_signal_t,
                               hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_store_screlease, hsa_signal_t,
                               hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_wait_scacquire, hsa_signal_t,
                               hsa_signal_condition_t, hsa_signal_value_t,
                               uint64_t, hsa_wait_state_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_subtract_screlease, hsa_signal_t,
                               hsa_signal_value_t)

// Memory management (core)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_register, void*, size_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_deregister, void*, size_t)

// Executable and code object management
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_create_alt, hsa_profile_t,
                               hsa_default_float_rounding_mode_t,
                               const char*, hsa_executable_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_destroy, hsa_executable_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_freeze, hsa_executable_t,
                               const char*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_get_symbol_by_name,
                               hsa_executable_t, const char*,
                               const hsa_agent_t*, hsa_executable_symbol_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_symbol_get_info,
                               hsa_executable_symbol_t,
                               hsa_executable_symbol_info_t, void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_code_object_reader_create_from_memory,
                               const void*, size_t, hsa_code_object_reader_t*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_code_object_reader_destroy,
                               hsa_code_object_reader_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_load_agent_code_object,
                               hsa_executable_t, hsa_agent_t,
                               hsa_code_object_reader_t, const char*,
                               hsa_loaded_code_object_t*)

//===----------------------------------------------------------------------===//
// HSA AMD Extension symbols
//===----------------------------------------------------------------------===//

// AMD memory pool management
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_agent_iterate_memory_pools, hsa_agent_t,
                               hsa_status_t (*)(hsa_amd_memory_pool_t, void*),
                               void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_get_info,
                               hsa_amd_memory_pool_t,
                               hsa_amd_memory_pool_info_t, void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_allocate,
                               hsa_amd_memory_pool_t, size_t, uint32_t, void**)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_free, void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_agents_allow_access, uint32_t,
                               const hsa_agent_t*, const uint32_t*, const void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_lock, void*, size_t,
                               hsa_agent_t*, int, void**)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_unlock, void*)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_fill, void*, uint32_t, size_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_async_copy, void*, hsa_agent_t,
                               const void*, hsa_agent_t, size_t, uint32_t,
                               const hsa_signal_t*, hsa_signal_t)

// AMD agent info
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_agent_memory_pool_get_info, hsa_agent_t,
                               hsa_amd_memory_pool_t,
                               hsa_amd_agent_memory_pool_info_t, void*)

// AMD profiling
IREE_HAL_HSA_OPTIONAL_PFN_DECL(hsa_amd_profiling_set_profiler_enabled,
                               hsa_queue_t*, int)
IREE_HAL_HSA_OPTIONAL_PFN_DECL(hsa_amd_profiling_get_dispatch_time,
                               hsa_agent_t, hsa_signal_t,
                               hsa_amd_profiling_dispatch_time_t*)
IREE_HAL_HSA_OPTIONAL_PFN_DECL(hsa_amd_profiling_get_async_copy_time,
                               hsa_signal_t,
                               hsa_amd_profiling_async_copy_time_t*)

// AMD signal operations
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_signal_create, hsa_signal_value_t,
                               uint32_t, const hsa_agent_t*, uint64_t,
                               hsa_signal_t*)

