// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_MPI_CHANNEL_PROVIDER_H_
#define IREE_HAL_UTILS_MPI_CHANNEL_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_mpi_dynamic_symbols_t iree_hal_mpi_dynamic_symbols_t;

// Returns true if MPI support has been configured by the user and should be
// used. This checks for required MPI environment variables that are set
// when running under mpirun.
IREE_API_EXPORT bool iree_hal_mpi_is_configured(void);

// Creates an MPI-based collective channel provider.
// On creation the provider will initialize MPI with MPI_Init and on destruction
// will finalize MPI with MPI_Finalize after which time MPI can never be used in
// the process again. Users should create one provider and share it across all
// devices used within the process.
IREE_API_EXPORT iree_status_t iree_hal_mpi_channel_provider_create(
    iree_allocator_t host_allocator,
    iree_hal_channel_provider_t** out_channel_provider);

// Returns true if |channel_provider| is an MPI channel provider.
IREE_API_EXPORT bool iree_hal_mpi_channel_provider_isa(
    iree_hal_channel_provider_t* channel_provider);

// Returns the dynamically loaded MPI symbols or NULL if |channel_provider| is
// not an MPI channel provider.
IREE_API_EXPORT iree_hal_mpi_dynamic_symbols_t*
iree_hal_mpi_channel_provider_symbols(
    iree_hal_channel_provider_t* channel_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_MPI_CHANNEL_PROVIDER_H_
