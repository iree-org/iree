// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// These declarations come from the ROCR-Runtime public headers:
// https://github.com/ROCm/ROCR-Runtime/tree/amd-staging/runtime/hsa-runtime/inc
//
// Specifically:
// https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/runtime/hsa-runtime/inc/hsa.h
// https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/runtime/hsa-runtime/inc/hsa_ext_amd.h
// https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/runtime/hsa-runtime/inc/hsa_ven_amd_loader.h

// TODO(benvanik): prune this to what we actually use - it's quite exhaustive
// today to reduce friction during bringup.

//===----------------------------------------------------------------------===//
// Library/System Management
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_init, DECL(), ARGS())
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_shut_down, DECL(),
                           ARGS())

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_register_system_event_handler,
                           DECL(hsa_amd_system_event_callback_t callback,
                                void* data),
                           ARGS(callback, data))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_system_get_info,
                           DECL(hsa_system_info_t attribute, void* value),
                           ARGS(attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_system_major_extension_supported,
                           DECL(uint16_t extension, uint16_t version_major,
                                uint16_t* version_minor, bool* result),
                           ARGS(extension, version_major, version_minor,
                                result))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_system_get_major_extension_table,
                           DECL(uint16_t extension, uint16_t version_major,
                                size_t table_length, void* table),
                           ARGS(extension, version_major, table_length, table))

//===----------------------------------------------------------------------===//
// Agents
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_agent_get_info,
                           DECL(hsa_agent_t agent, hsa_agent_info_t attribute,
                                void* value),
                           ARGS(agent, attribute, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_iterate_agents,
    DECL(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data),
    ARGS(callback, data))

//===----------------------------------------------------------------------===//
// Memory
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_cache_get_info,
                           DECL(hsa_cache_t cache, hsa_cache_info_t attribute,
                                void* value),
                           ARGS(cache, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_agent_iterate_caches,
    DECL(hsa_agent_t agent,
         hsa_status_t (*callback)(hsa_cache_t cache, void* data), void* data),
    ARGS(agent, callback, data))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_region_get_info,
                           DECL(hsa_region_t region,
                                hsa_region_info_t attribute, void* value),
                           ARGS(region, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_agent_iterate_regions,
    DECL(hsa_agent_t agent,
         hsa_status_t (*callback)(hsa_region_t region, void* data), void* data),
    ARGS(agent, callback, data))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_memory_pool_get_info,
                           DECL(hsa_amd_memory_pool_t memory_pool,
                                hsa_amd_memory_pool_info_t attribute,
                                void* value),
                           ARGS(memory_pool, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_agent_iterate_memory_pools,
    DECL(hsa_agent_t agent,
         hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool,
                                  void* data),
         void* data),
    ARGS(agent, callback, data))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_agent_memory_pool_get_info,
    DECL(hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
         hsa_amd_agent_memory_pool_info_t attribute, void* value),
    ARGS(agent, memory_pool, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_agents_allow_access,
                           DECL(uint32_t num_agents, const hsa_agent_t* agents,
                                const uint32_t* flags, const void* ptr),
                           ARGS(num_agents, agents, flags, ptr))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_memory_copy,
                           DECL(void* dst, const void* src, size_t size),
                           ARGS(dst, src, size))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_memory_pool_allocate,
                           DECL(hsa_amd_memory_pool_t memory_pool, size_t size,
                                uint32_t flags, void** ptr),
                           ARGS(memory_pool, size, flags, ptr))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_memory_pool_free,
                           DECL(void* ptr), ARGS(ptr))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_memory_lock_to_pool,
    DECL(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent,
         hsa_amd_memory_pool_t pool, uint32_t flags, void** agent_ptr),
    ARGS(host_ptr, size, agents, num_agent, pool, flags, agent_ptr))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_memory_unlock,
                           DECL(void* host_ptr), ARGS(host_ptr))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_interop_map_buffer,
                           DECL(uint32_t num_agents, hsa_agent_t* agents,
                                int interop_handle, uint32_t flags,
                                size_t* size, void** ptr, size_t* metadata_size,
                                const void** metadata),
                           ARGS(num_agents, agents, interop_handle, flags, size,
                                ptr, metadata_size, metadata))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_interop_unmap_buffer, DECL(void* ptr),
                           ARGS(ptr))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_pointer_info,
    DECL(const void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
         uint32_t* num_agents_accessible, hsa_agent_t** accessible),
    ARGS(ptr, info, alloc, num_agents_accessible, accessible))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_ipc_memory_create,
                           DECL(void* ptr, size_t len,
                                hsa_amd_ipc_memory_t* handle),
                           ARGS(ptr, len, handle))
IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_ipc_memory_attach,
    DECL(const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
         const hsa_agent_t* mapping_agents, void** mapped_ptr),
    ARGS(handle, len, num_agents, mapping_agents, mapped_ptr))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_ipc_memory_detach, DECL(void* mapped_ptr),
                           ARGS(mapped_ptr))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_portable_export_dmabuf,
                           DECL(const void* ptr, size_t size, int* dmabuf,
                                uint64_t* offset),
                           ARGS(ptr, size, dmabuf, offset))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_portable_close_dmabuf, DECL(int dmabuf),
                           ARGS(dmabuf))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_svm_attributes_get,
                           DECL(void* ptr, size_t size,
                                hsa_amd_svm_attribute_pair_t* attribute_list,
                                size_t attribute_count),
                           ARGS(ptr, size, attribute_list, attribute_count))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_svm_attributes_set,
                           DECL(void* ptr, size_t size,
                                hsa_amd_svm_attribute_pair_t* attribute_list,
                                size_t attribute_count),
                           ARGS(ptr, size, attribute_list, attribute_count))
IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_amd_svm_prefetch_async,
    DECL(void* ptr, size_t size, hsa_agent_t agent, uint32_t num_dep_signals,
         const hsa_signal_t* dep_signals, hsa_signal_t completion_signal),
    ARGS(ptr, size, agent, num_dep_signals, dep_signals, completion_signal))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_vmem_address_reserve_align,
                           DECL(void** va, size_t size, uint64_t address,
                                uint64_t alignment, uint64_t flags),
                           ARGS(va, size, address, alignment, flags))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_vmem_address_free,
                           DECL(void* va, size_t size), ARGS(va, size))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_vmem_set_access,
                           DECL(void* va, size_t size,
                                const hsa_amd_memory_access_desc_t* desc,
                                size_t desc_cnt),
                           ARGS(va, size, desc, desc_cnt))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_vmem_handle_create,
                           DECL(hsa_amd_memory_pool_t pool, size_t size,
                                hsa_amd_memory_type_t type, uint64_t flags,
                                hsa_amd_vmem_alloc_handle_t* memory_handle),
                           ARGS(pool, size, type, flags, memory_handle))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_vmem_handle_release,
                           DECL(hsa_amd_vmem_alloc_handle_t memory_handle),
                           ARGS(memory_handle))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_vmem_map,
                           DECL(void* va, size_t size, size_t in_offset,
                                hsa_amd_vmem_alloc_handle_t memory_handle,
                                uint64_t flags),
                           ARGS(va, size, in_offset, memory_handle, flags))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_vmem_unmap,
                           DECL(void* va, size_t size), ARGS(va, size))

//===----------------------------------------------------------------------===//
// Signals
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_signal_create,
                           DECL(hsa_signal_value_t initial_value,
                                uint32_t num_consumers,
                                const hsa_agent_t* consumers,
                                uint64_t attributes, hsa_signal_t* signal),
                           ARGS(initial_value, num_consumers, consumers,
                                attributes, signal))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_signal_destroy,
                           DECL(hsa_signal_t signal), ARGS(signal))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_load_scacquire, DECL(hsa_signal_t signal),
                           ARGS(signal))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_load_relaxed, DECL(hsa_signal_t signal),
                           ARGS(signal))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_store_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_store_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_silent_store_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void,
                           hsa_signal_silent_store_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_exchange_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_exchange_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_exchange_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_exchange_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_cas_scacq_screl,
                           DECL(hsa_signal_t signal,
                                hsa_signal_value_t expected,
                                hsa_signal_value_t value),
                           ARGS(signal, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_cas_scacquire,
                           DECL(hsa_signal_t signal,
                                hsa_signal_value_t expected,
                                hsa_signal_value_t value),
                           ARGS(signal, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_cas_relaxed,
                           DECL(hsa_signal_t signal,
                                hsa_signal_value_t expected,
                                hsa_signal_value_t value),
                           ARGS(signal, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, hsa_signal_value_t,
                           hsa_signal_cas_screlease,
                           DECL(hsa_signal_t signal,
                                hsa_signal_value_t expected,
                                hsa_signal_value_t value),
                           ARGS(signal, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_add_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_add_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_add_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_add_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_subtract_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_subtract_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_subtract_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_subtract_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_and_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_and_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_and_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_and_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_or_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_or_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_or_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_or_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_xor_scacq_screl,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_xor_scacquire,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_xor_relaxed,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_SIGNALS, void, hsa_signal_xor_screlease,
                           DECL(hsa_signal_t signal, hsa_signal_value_t value),
                           ARGS(signal, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_signal_value_t, hsa_signal_wait_scacquire,
    DECL(hsa_signal_t signal, hsa_signal_condition_t condition,
         hsa_signal_value_t compare_value, uint64_t timeout_hint,
         hsa_wait_state_t wait_state_hint),
    ARGS(signal, condition, compare_value, timeout_hint, wait_state_hint))
IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_signal_value_t, hsa_signal_wait_relaxed,
    DECL(hsa_signal_t signal, hsa_signal_condition_t condition,
         hsa_signal_value_t compare_value, uint64_t timeout_hint,
         hsa_wait_state_t wait_state_hint),
    ARGS(signal, condition, compare_value, timeout_hint, wait_state_hint))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, uint32_t, hsa_amd_signal_wait_any,
                           DECL(uint32_t signal_count, hsa_signal_t* signals,
                                hsa_signal_condition_t* conds,
                                hsa_signal_value_t* values,
                                uint64_t timeout_hint,
                                hsa_wait_state_t wait_hint,
                                hsa_signal_value_t* satisfying_value),
                           ARGS(signal_count, signals, conds, values,
                                timeout_hint, wait_hint, satisfying_value))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_ipc_signal_create,
                           DECL(hsa_signal_t signal,
                                hsa_amd_ipc_signal_t* handle),
                           ARGS(signal, handle))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_ipc_signal_attach,
                           DECL(const hsa_amd_ipc_signal_t* handle,
                                hsa_signal_t* signal),
                           ARGS(handle, signal))

//===----------------------------------------------------------------------===//
// Queues
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_queue_create,
    DECL(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
         void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
         void* data, uint32_t private_segment_size, uint32_t group_segment_size,
         hsa_queue_t** queue),
    ARGS(agent, size, type, callback, data, private_segment_size,
         group_segment_size, queue))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_soft_queue_create,
    DECL(hsa_region_t region, uint32_t size, hsa_queue_type32_t type,
         uint32_t features, hsa_signal_t doorbell_signal, hsa_queue_t** queue),
    ARGS(region, size, type, features, doorbell_signal, queue))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_queue_destroy,
                           DECL(hsa_queue_t* queue), ARGS(queue))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_queue_inactivate,
                           DECL(hsa_queue_t* queue), ARGS(queue))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_profiling_set_profiler_enabled,
                           DECL(hsa_queue_t* queue, int enable),
                           ARGS(queue, enable))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_amd_queue_get_info,
                           DECL(hsa_queue_t* queue,
                                hsa_queue_info_attribute_t attribute,
                                void* value),
                           ARGS(queue, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_queue_cu_set_mask,
                           DECL(const hsa_queue_t* queue,
                                uint32_t num_cu_mask_count,
                                const uint32_t* cu_mask),
                           ARGS(queue, num_cu_mask_count, cu_mask))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_amd_queue_set_priority,
                           DECL(hsa_queue_t* queue,
                                hsa_amd_queue_priority_t priority),
                           ARGS(queue, priority))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_load_read_index_scacquire,
                           DECL(const hsa_queue_t* queue), ARGS(queue))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_load_read_index_relaxed,
                           DECL(const hsa_queue_t* queue), ARGS(queue))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_load_write_index_relaxed,
                           DECL(const hsa_queue_t* queue), ARGS(queue))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_load_write_index_scacquire,
                           DECL(const hsa_queue_t* queue), ARGS(queue))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, void,
                           hsa_queue_store_write_index_relaxed,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, void,
                           hsa_queue_store_write_index_screlease,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_cas_write_index_scacq_screl,
                           DECL(const hsa_queue_t* queue, uint64_t expected,
                                uint64_t value),
                           ARGS(queue, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_cas_write_index_scacquire,
                           DECL(const hsa_queue_t* queue, uint64_t expected,
                                uint64_t value),
                           ARGS(queue, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_cas_write_index_relaxed,
                           DECL(const hsa_queue_t* queue, uint64_t expected,
                                uint64_t value),
                           ARGS(queue, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_cas_write_index_screlease,
                           DECL(const hsa_queue_t* queue, uint64_t expected,
                                uint64_t value),
                           ARGS(queue, expected, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_add_write_index_scacq_screl,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_add_write_index_scacquire,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_add_write_index_relaxed,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, uint64_t,
                           hsa_queue_add_write_index_screlease,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, void,
                           hsa_queue_store_read_index_relaxed,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_QUEUES, void,
                           hsa_queue_store_read_index_screlease,
                           DECL(const hsa_queue_t* queue, uint64_t value),
                           ARGS(queue, value))

//===----------------------------------------------------------------------===//
// Executables
//===----------------------------------------------------------------------===//

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_agent_iterate_isas,
    DECL(hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t isa, void* data),
         void* data),
    ARGS(agent, callback, data))
IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_isa_get_info_alt,
                           DECL(hsa_isa_t isa, hsa_isa_info_t attribute,
                                void* value),
                           ARGS(isa, attribute, value))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_code_object_reader_create_from_memory,
                           DECL(const void* code_object, size_t size,
                                hsa_code_object_reader_t* code_object_reader),
                           ARGS(code_object, size, code_object_reader))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_code_object_reader_destroy,
                           DECL(hsa_code_object_reader_t code_object_reader),
                           ARGS(code_object_reader))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_executable_create_alt,
    DECL(hsa_profile_t profile,
         hsa_default_float_rounding_mode_t default_float_rounding_mode,
         const char* options, hsa_executable_t* executable),
    ARGS(profile, default_float_rounding_mode, options, executable))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_executable_destroy,
                           DECL(hsa_executable_t executable), ARGS(executable))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_executable_load_agent_code_object,
    DECL(hsa_executable_t executable, hsa_agent_t agent,
         hsa_code_object_reader_t code_object_reader, const char* options,
         hsa_loaded_code_object_t* loaded_code_object),
    ARGS(executable, agent, code_object_reader, options, loaded_code_object))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t, hsa_executable_freeze,
                           DECL(hsa_executable_t executable,
                                const char* options),
                           ARGS(executable, options))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_executable_validate_alt,
                           DECL(hsa_executable_t executable,
                                const char* options, uint32_t* result),
                           ARGS(executable, options, result))

IREE_HAL_AMDGPU_LIBHSA_PFN(
    TRACE_ALWAYS, hsa_status_t, hsa_executable_get_symbol_by_name,
    DECL(hsa_executable_t executable, const char* symbol_name,
         const hsa_agent_t* agent, hsa_executable_symbol_t* symbol),
    ARGS(executable, symbol_name, agent, symbol))

IREE_HAL_AMDGPU_LIBHSA_PFN(TRACE_ALWAYS, hsa_status_t,
                           hsa_executable_symbol_get_info,
                           DECL(hsa_executable_symbol_t executable_symbol,
                                hsa_executable_symbol_info_t attribute,
                                void* value),
                           ARGS(executable_symbol, attribute, value))

#undef IREE_HAL_AMDGPU_LIBHSA_PFN
#undef DECL
#undef ARGS
