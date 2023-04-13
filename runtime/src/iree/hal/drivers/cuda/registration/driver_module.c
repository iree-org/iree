// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/api.h"
#include "iree/hal/drivers/cuda/cuda_device.h"
#include "iree/hal/drivers/cuda/status_util.h"

// Force using CUDA streams until we support command buffer caching to avoid the
// overhead of graph creation.
IREE_FLAG(
    bool, cuda_use_streams, true,
    "Use CUDA streams for executing command buffers (instead of graphs).");

IREE_FLAG(bool, cuda_allow_inline_execution, false,
          "Allow command buffers to execute inline against CUDA streams when "
          "possible.");

IREE_FLAG(bool, cuda_tracing, true,
          "Enables tracing of stream events when Tracy instrumentation is "
          "enabled. Severely impacts benchmark timings and should only be used "
          "when analyzing dispatch timings.");

IREE_FLAG(int32_t, cuda_default_index, 0, "Index of the default CUDA device.");

IREE_FLAG(int32_t, cuda_nccl_default_rank, 0,
          "Participant rank within the default collective group.");
IREE_FLAG(int32_t, cuda_nccl_default_count, 0,
          "Participant count of the default collective group");

// Default implementation of the collective channel provider that just uses the
// NCCL_COMM_ID environment variable for configuration. Hosting layers would
// want to use their own implementation to exchange IDs.
static iree_status_t iree_hal_cuda_nccl_query_group_params(
    void* self, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_byte_span_t id_storage,
    iree_hal_channel_params_t* params) {
  IREE_ASSERT_EQ(id_storage.data_length, sizeof(iree_hal_cuda_nccl_id_t));

  iree_hal_cuda_dynamic_symbols_t* syms =
      iree_hal_cuda_get_dynamic_symbols(device);

  if (!syms->mpi_library) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "MPI should be loaded to use NCCL collective operations.");
  }

  // Until we have multi channel support, we only creates the default channel.
  IREE_ASSERT_EQ(params->rank, IREE_HAL_CHANNEL_RANK_DEFAULT);
  IREE_ASSERT_EQ(params->count, IREE_HAL_CHANNEL_COUNT_DEFAULT);

  // Update the rank and count.
  MPI_RETURN_IF_ERROR(syms,
                      MPI_Comm_rank(syms->ompi_mpi_comm_world, &params->rank),
                      "MPI_Comm_rank");
  MPI_RETURN_IF_ERROR(syms,
                      MPI_Comm_size(syms->ompi_mpi_comm_world, &params->count),
                      "MPI_Comm_size");
  ;

  iree_hal_cuda_nccl_id_t* id = (iree_hal_cuda_nccl_id_t*)id_storage.data;
  if (params->rank == 0) {
    // The root process of the group creates the unique ID and broadcasts it
    // to the others.
    IREE_RETURN_IF_ERROR(iree_hal_cuda_nccl_get_unique_id(device, id));
  }

  MPI_RETURN_IF_ERROR(syms,
                      MPI_Bcast(id, id_storage.data_length, syms->ompi_mpi_byte,
                                0, syms->ompi_mpi_comm_world),
                      "MPI_Bcast");
  params->id = iree_const_cast_byte_span(id_storage);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // NOTE: we could query supported cuda versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = iree_string_view_literal("cuda"),
      .full_name = iree_string_view_literal("CUDA (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (!iree_string_view_equal(driver_name, IREE_SV("cuda"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_device_params_t default_params;
  iree_hal_cuda_device_params_initialize(&default_params);
  if (FLAG_cuda_use_streams) {
    default_params.command_buffer_mode =
        IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  }
  default_params.allow_inline_execution = FLAG_cuda_allow_inline_execution;
  default_params.stream_tracing = FLAG_cuda_tracing;
  default_params.channel_provider = (iree_hal_channel_provider_t){
      .self = NULL,
      .query_group_params = iree_hal_cuda_nccl_query_group_params,
  };

  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);
  driver_options.default_device_index = FLAG_cuda_default_index;
  driver_options.default_rank = FLAG_cuda_nccl_default_rank;
  driver_options.default_count = FLAG_cuda_nccl_default_count;

  iree_status_t status =
      iree_hal_cuda_driver_create(driver_name, &default_params, &driver_options,
                                  host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda_driver_factory_enumerate,
      .try_create = iree_hal_cuda_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
