// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/mpi_channel_provider.h"

#include <stdlib.h>

#include "iree/base/tracing.h"
#include "iree/hal/utils/libmpi.h"

// Returns true if |var_name| is set to a non-empty value in the environment.
static bool iree_hal_mpi_env_is_set(const char* var_name) {
  const char* var_value = getenv(var_name);
  return var_value && strlen(var_value) > 0;
}

// For now this simply checks that the world size is set. We could verify
// more of the environment but if a user is partially configuring things
// manually YMMV.
IREE_API_EXPORT bool iree_hal_mpi_is_configured(void) {
  // TODO: find a better approach that is more portable across implementations.
  // PMI_RANK/PMI_SIZE seem common ones (MS/Intel/Slurm) but OpenMPI uses their
  // own.
  return iree_hal_mpi_env_is_set("PMI_SIZE") ||
         iree_hal_mpi_env_is_set("OMPI_COMM_WORLD_SIZE") ||
         iree_hal_mpi_env_is_set("MPIEXEC_HOSTNAME");
}

typedef struct iree_hal_mpi_channel_provider_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_dynamic_library_t* library;
  iree_hal_mpi_dynamic_symbols_t symbols;
} iree_hal_mpi_channel_provider_t;

static const iree_hal_channel_provider_vtable_t
    iree_hal_mpi_channel_provider_vtable;

static iree_hal_mpi_channel_provider_t* iree_hal_mpi_channel_provider_cast(
    iree_hal_channel_provider_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_mpi_channel_provider_vtable);
  return (iree_hal_mpi_channel_provider_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_mpi_channel_provider_create(
    iree_allocator_t host_allocator,
    iree_hal_channel_provider_t** out_channel_provider) {
  IREE_ASSERT_ARGUMENT(out_channel_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_mpi_channel_provider_t* channel_provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*channel_provider),
                                (void**)&channel_provider));
  iree_hal_resource_initialize(&iree_hal_mpi_channel_provider_vtable,
                               &channel_provider->resource);
  channel_provider->host_allocator = host_allocator;

  // Attempt to load the shared library. This will fail if it's not found,
  // not compatible with the process (wrong arch), or missing symbols (out of
  // date).
  iree_status_t status = iree_hal_mpi_library_load(
      host_allocator, &channel_provider->library, &channel_provider->symbols);

  // If the library successfully loaded then try to initialize MPI.
  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "MPI_Init");
    status = MPI_RESULT_TO_STATUS(&channel_provider->symbols,
                                  MPI_Init(NULL, NULL), "MPI_Init");
    IREE_TRACE_ZONE_END(z1);
  }

  if (iree_status_is_ok(status)) {
    *out_channel_provider = (iree_hal_channel_provider_t*)channel_provider;
  } else {
    iree_hal_channel_provider_release(
        (iree_hal_channel_provider_t*)channel_provider);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns true if MPI has been initialized.
static bool iree_hal_mpi_channel_provider_is_initialized(
    iree_hal_mpi_channel_provider_t* channel_provider) {
  if (!channel_provider->library) return false;
  int flag = 0;
  MPI_IGNORE_ERROR(&channel_provider->symbols, MPI_Initialized(&flag));
  return flag ? true : false;
}

static void iree_hal_mpi_channel_provider_destroy(
    iree_hal_channel_provider_t* base_channel_provider) {
  iree_hal_mpi_channel_provider_t* channel_provider =
      iree_hal_mpi_channel_provider_cast(base_channel_provider);
  iree_allocator_t host_allocator = channel_provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // We must finalize MPI before unloading the library.
  // NOTE: once finalized MPI can never be used in the process again!
  if (iree_hal_mpi_channel_provider_is_initialized(channel_provider)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "MPI_Finalize");
    MPI_IGNORE_ERROR(&channel_provider->symbols, MPI_Finalize());
    IREE_TRACE_ZONE_END(z1);
  }

  // Reset the symbols in case anyone is hanging on to them. ASAN should
  // complain anyway but it's not available everywhere.
  memset(&channel_provider->symbols, 0, sizeof(channel_provider->symbols));

  iree_dynamic_library_release(channel_provider->library);

  iree_allocator_free(host_allocator, channel_provider);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT bool iree_hal_mpi_channel_provider_isa(
    iree_hal_channel_provider_t* channel_provider) {
  return iree_hal_resource_is(channel_provider,
                              &iree_hal_mpi_channel_provider_vtable);
}

IREE_API_EXPORT iree_hal_mpi_dynamic_symbols_t*
iree_hal_mpi_channel_provider_symbols(
    iree_hal_channel_provider_t* base_channel_provider) {
  IREE_ASSERT_ARGUMENT(base_channel_provider);
  if (!iree_hal_mpi_channel_provider_isa(base_channel_provider)) return NULL;
  iree_hal_mpi_channel_provider_t* channel_provider =
      iree_hal_mpi_channel_provider_cast(base_channel_provider);
  return &channel_provider->symbols;
}

static iree_status_t iree_hal_mpi_channel_provider_query_default_rank_and_count(
    iree_hal_channel_provider_t* base_channel_provider, int32_t* out_rank,
    int32_t* out_count) {
  iree_hal_mpi_channel_provider_t* channel_provider =
      iree_hal_mpi_channel_provider_cast(base_channel_provider);

  static_assert(sizeof(int32_t) == sizeof(int), "MPI uses int");
  MPI_RETURN_IF_ERROR(
      &channel_provider->symbols,
      MPI_Comm_rank(IREE_MPI_COMM_WORLD(&channel_provider->symbols),
                    (int*)out_rank),
      "MPI_Comm_rank");
  MPI_RETURN_IF_ERROR(
      &channel_provider->symbols,
      MPI_Comm_size(IREE_MPI_COMM_WORLD(&channel_provider->symbols),
                    (int*)out_count),
      "MPI_Comm_size");

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_mpi_channel_provider_exchange_default_id(
    iree_hal_channel_provider_t* base_channel_provider, iree_byte_span_t id) {
  iree_hal_mpi_channel_provider_t* channel_provider =
      iree_hal_mpi_channel_provider_cast(base_channel_provider);

  // Exchange the ID with all other participants. The root participant will
  // send its ID while the others will receive it.
  MPI_RETURN_IF_ERROR(
      &channel_provider->symbols,
      MPI_Bcast(id.data, id.data_length,
                IREE_MPI_BYTE(&channel_provider->symbols), 0,
                IREE_MPI_COMM_WORLD(&channel_provider->symbols)),
      "MPI_Bcast");

  return iree_ok_status();
}

static const iree_hal_channel_provider_vtable_t
    iree_hal_mpi_channel_provider_vtable = {
        .destroy = iree_hal_mpi_channel_provider_destroy,
        .query_default_rank_and_count =
            iree_hal_mpi_channel_provider_query_default_rank_and_count,
        .exchange_default_id =
            iree_hal_mpi_channel_provider_exchange_default_id,
};
