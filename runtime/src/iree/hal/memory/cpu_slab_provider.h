// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_
#define IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a slab provider that allocates host memory via |host_allocator|
// (typically the system allocator / malloc). Slabs are plain host memory
// with no special alignment, registration, or NUMA affinity.
//
// Reports memory type HOST_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
// and supports all buffer usage flags (TRANSFER, DISPATCH, MAPPING, etc.).
//
// This is the simplest slab provider - intended for CPU-only testing and as
// the backing for pass-through pools on host targets.
IREE_API_EXPORT iree_status_t iree_hal_cpu_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_
