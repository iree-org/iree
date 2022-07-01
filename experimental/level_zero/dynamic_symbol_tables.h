// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/* Context/Device/Runtime related APIs */
ZE_PFN_DECL(zeInit, ze_init_flags_t)
ZE_PFN_DECL(zeContextCreate, ze_driver_handle_t, const ze_context_desc_t *,
            ze_context_handle_t *)
ZE_PFN_DECL(zeContextDestroy, ze_context_handle_t)
ZE_PFN_DECL(zeDriverGet, uint32_t *,
            ze_driver_handle_t *)  // No direct, need to modify
ZE_PFN_DECL(zeDeviceGet, ze_driver_handle_t, uint32_t *,
            ze_device_handle_t *)  // Can get device handle and count.
ZE_PFN_DECL(zeDeviceGetProperties, ze_device_handle_t,
            ze_device_properties_t *)  // Can get name.

/* Command Buffer/List related APIs*/
ZE_PFN_DECL(zeDeviceGetCommandQueueGroupProperties, ze_device_handle_t,
            uint32_t *, ze_command_queue_group_properties_t *)
ZE_PFN_DECL(zeCommandQueueCreate, ze_context_handle_t, ze_device_handle_t,
            const ze_command_queue_desc_t *, ze_command_queue_handle_t *)
ZE_PFN_DECL(zeCommandListCreate, ze_context_handle_t, ze_device_handle_t,
            const ze_command_list_desc_t *, ze_command_list_handle_t *)
ZE_PFN_DECL(zeCommandListAppendLaunchKernel, ze_command_list_handle_t,
            ze_kernel_handle_t, const ze_group_count_t *, ze_event_handle_t,
            uint32_t, ze_event_handle_t *)
ZE_PFN_DECL(zeCommandListClose, ze_command_list_handle_t)
ZE_PFN_DECL(zeCommandQueueExecuteCommandLists, ze_command_queue_handle_t,
            uint32_t, ze_command_list_handle_t *, ze_fence_handle_t)
ZE_PFN_DECL(zeCommandQueueSynchronize, ze_command_queue_handle_t, uint64_t)
ZE_PFN_DECL(zeCommandListDestroy, ze_command_list_handle_t)
ZE_PFN_DECL(zeCommandQueueDestroy, ze_command_queue_handle_t)

/* Memory related APIs*/
// NOTE: Asynchronous/Blocking set in flags.
// NOTE: Intel Shared Memory == Unified/Managed memory, where memory is shared
// between host and devices.
ZE_PFN_DECL(zeCommandListAppendMemoryFill, ze_command_list_handle_t, void *,
            const void *, size_t, size_t, ze_event_handle_t, uint32_t,
            ze_event_handle_t *)
ZE_PFN_DECL(zeCommandListAppendMemoryCopy, ze_command_list_handle_t, void *,
            const void *, size_t, ze_event_handle_t, uint32_t,
            ze_event_handle_t *)
ZE_PFN_DECL(zeMemAllocDevice, ze_context_handle_t,
            const ze_device_mem_alloc_desc_t *, size_t, size_t,
            ze_device_handle_t, void **)
ZE_PFN_DECL(zeMemAllocShared, ze_context_handle_t,
            const ze_device_mem_alloc_desc_t *,
            const ze_host_mem_alloc_desc_t *, size_t, size_t,
            ze_device_handle_t, void **)
ZE_PFN_DECL(zeMemAllocHost, ze_context_handle_t,
            const ze_host_mem_alloc_desc_t *, size_t, size_t, void **)
ZE_PFN_DECL(zeMemFree, ze_context_handle_t, void *)
ZE_PFN_DECL(zeCommandListAppendBarrier, ze_command_list_handle_t,
            ze_event_handle_t, uint32_t, ze_event_handle_t *)
ZE_PFN_DECL(zeEventPoolCreate, ze_context_handle_t, const ze_event_pool_desc_t *, uint32_t, ze_device_handle_t *, ze_event_pool_handle_t *)
ZE_PFN_DECL(zeEventCreate, ze_event_pool_handle_t, const ze_event_desc_t *, ze_event_handle_t *)
ZE_PFN_DECL(zeEventDestroy, ze_event_handle_t)
ZE_PFN_DECL(zeEventPoolDestroy, ze_event_pool_handle_t)
ZE_PFN_DECL(zeCommandListAppendSignalEvent, ze_command_list_handle_t, ze_event_handle_t)
ZE_PFN_DECL(zeCommandListAppendWaitOnEvents, ze_command_list_handle_t, uint32_t, ze_event_handle_t*)
ZE_PFN_DECL(zeCommandListAppendEventReset, ze_command_list_handle_t, ze_event_handle_t)

/* Kernel generation related APIs*/
ZE_PFN_DECL(zeModuleCreate, ze_context_handle_t, ze_device_handle_t,
            const ze_module_desc_t *, ze_module_handle_t *,
            ze_module_build_log_handle_t *)
ZE_PFN_DECL(zeKernelCreate, ze_module_handle_t, const ze_kernel_desc_t *,
            ze_kernel_handle_t *)
ZE_PFN_DECL(zeKernelSuggestGroupSize, ze_kernel_handle_t, uint32_t, uint32_t,
            uint32_t, uint32_t *, uint32_t *, uint32_t *)
ZE_PFN_DECL(zeKernelSetGroupSize, ze_kernel_handle_t, uint32_t, uint32_t,
            uint32_t)
ZE_PFN_DECL(zeKernelSetArgumentValue, ze_kernel_handle_t, uint32_t, size_t,
            const void *)
ZE_PFN_DECL(zeModuleBuildLogDestroy, ze_module_build_log_handle_t)
ZE_PFN_DECL(zeModuleBuildLogGetString, ze_module_build_log_handle_t, size_t *,
            char *)