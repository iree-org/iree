// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/cuda/client.h"

namespace iree::pjrt::cuda {

CUDAClientInstance::CUDAClientInstance(std::unique_ptr<Platform> platform)
    : ClientInstance(std::move(platform)) {
  // Seems that it must match how registered. Action at a distance not
  // great.
  // TODO: Get this when constructing the client so it is guaranteed to
  // match.
  cached_platform_name_ = "iree_cuda";
}

CUDAClientInstance::~CUDAClientInstance() {}

iree_status_t CUDAClientInstance::CreateDriver(iree_hal_driver_t** out_driver) {
  iree_string_view_t driver_name = iree_make_cstring_view("cuda");

  // Device params.
  // TODO: Plumb through some important params:
  //   nccl_default_id
  //   nccl_default_rank
  //   nccl_default_count
  // Switch command_buffer_mode to graphs when ready.
  iree_hal_cuda_device_params_t default_params;
  iree_hal_cuda_device_params_initialize(&default_params);
  default_params.command_buffer_mode = IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  default_params.allow_inline_execution = false;

  // Driver params.
  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);
  driver_options.default_device_index = 0;

  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_driver_create(driver_name, &default_params, &driver_options,
                                  host_allocator_, out_driver));
  logger().debug("CUDA driver created");
  return iree_ok_status();
}

bool CUDAClientInstance::SetDefaultCompilerFlags(CompilerJob* compiler_job) {
  return compiler_job->SetFlag("--iree-hal-target-backends=cuda");
}

}  // namespace iree::pjrt::cuda
