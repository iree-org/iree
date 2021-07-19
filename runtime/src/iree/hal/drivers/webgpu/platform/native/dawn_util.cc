// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// DO NOT SUBMIT
// #include <wgpu.h>  // wgpu-native implementation only

#include "iree/hal/drivers/webgpu/platform/webgpu.h"

extern "C" {

//===----------------------------------------------------------------------===//
// Implementation compatibility layer
//===----------------------------------------------------------------------===//

void iree_wgpuBindGroupDrop(WGPUBindGroup bindGroup) {
  // wgpuBindGroupDrop(bindGroup);
}

void iree_wgpuBindGroupLayoutDrop(WGPUBindGroupLayout bindGroupLayout) {
  // wgpuBindGroupLayoutDrop(bindGroupLayout);
}

void iree_wgpuBufferDrop(WGPUBuffer buffer) {
  // NOTE: drop != destroy (destroy is immediate) but it's all dawn has.
  // wgpuBufferDrop(buffer);
  wgpuBufferDestroy(buffer);
}

void iree_wgpuCommandBufferDrop(WGPUCommandBuffer commandBuffer) {
  // wgpuCommandBufferDrop(commandBuffer);
}

void iree_wgpuCommandEncoderDrop(WGPUCommandEncoder commandEncoder) {
  // wgpuCommandEncoderDrop(commandEncoder);
}

void iree_wgpuComputePipelineDrop(WGPUComputePipeline computePipeline) {
  // wgpuComputePipelineDrop(computePipeline);
}

void iree_wgpuPipelineLayoutDrop(WGPUPipelineLayout pipelineLayout) {
  // wgpuPipelineLayoutDrop(pipelineLayout);
}

void iree_wgpuQuerySetDrop(WGPUQuerySet querySet) {
  // wgpuQuerySetDrop(querySet);
}

void iree_wgpuShaderModuleDrop(WGPUShaderModule shaderModule) {
  // wgpuShaderModuleDrop(shaderModule);
}

//===----------------------------------------------------------------------===//
// Speculative WebGPU API additions
//===----------------------------------------------------------------------===//

static void iree_hal_webgpu_buffer_map_sync_callback(
    WGPUBufferMapAsyncStatus status, void* userdata) {
  IREEWGPUBufferMapSyncStatus* sync_status =
      (IREEWGPUBufferMapSyncStatus*)userdata;
  switch (status) {
    case WGPUBufferMapAsyncStatus_Success:
      *sync_status = IREEWGPUBufferMapSyncStatus_Success;
      break;
    case WGPUBufferMapAsyncStatus_Error:
      *sync_status = IREEWGPUBufferMapSyncStatus_Error;
      break;
    default:
    case WGPUBufferMapAsyncStatus_Unknown:
      *sync_status = IREEWGPUBufferMapSyncStatus_Unknown;
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      *sync_status = IREEWGPUBufferMapSyncStatus_DeviceLost;
      break;
  }
}

IREEWGPUBufferMapSyncStatus iree_wgpuBufferMapSync(WGPUDevice device,
                                                   WGPUBuffer buffer,
                                                   WGPUMapModeFlags mode,
                                                   size_t offset, size_t size) {
  IREEWGPUBufferMapSyncStatus status = IREEWGPUBufferMapSyncStatus_Unknown;
  wgpuBufferMapAsync(buffer, mode, offset, size,
                     iree_hal_webgpu_buffer_map_sync_callback, &status);
  // DO NOT SUBMIT
  // wgpuDevicePoll(device, /*force_wait=*/true);
  return status;
}

}  // extern "C"
