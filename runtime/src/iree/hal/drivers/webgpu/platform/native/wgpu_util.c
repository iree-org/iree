// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <wgpu.h>  // wgpu-native implementation only

#include "iree/hal/drivers/webgpu/platform/webgpu.h"

//===----------------------------------------------------------------------===//
// Implementation compatibility layer
//===----------------------------------------------------------------------===//

#define WGPU_NATIVE_INSTANCE ((WGPUInstance)((uintptr_t)0xABADD00Du))

WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const* descriptor) {
  // wgpu-native does not have instances and just uses globals for everything :(
  // We use a sentinel value here so that we can do null checks in places for
  // implementations that do use instances.
  return WGPU_NATIVE_INSTANCE;
}

void wgpuDeviceDestroy(WGPUDevice device) {
  // wgpu-native does not export this but does have a drop to deref.
  wgpuDeviceDrop(device);
}

void const* wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset,
                                          size_t size) {
  // wgpu-native doesn't have this, for some reason.
  return wgpuBufferGetMappedRange(buffer, offset, size);
}

void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder) {
  // No-op; wgpu-native does not export this symbol.
}

void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder,
                                      char const* groupLabel) {
  // No-op; wgpu-native does not export this symbol.
}

void iree_wgpuBindGroupDrop(WGPUBindGroup bindGroup) {
  wgpuBindGroupDrop(bindGroup);
}

void iree_wgpuBindGroupLayoutDrop(WGPUBindGroupLayout bindGroupLayout) {
  wgpuBindGroupLayoutDrop(bindGroupLayout);
}

void iree_wgpuBufferDrop(WGPUBuffer buffer) { wgpuBufferDrop(buffer); }

void iree_wgpuCommandBufferDrop(WGPUCommandBuffer commandBuffer) {
  // wgpu-native crashes if this method is called. I have no idea why - rust...
  // wgpuCommandBufferDrop(commandBuffer);
}

void iree_wgpuCommandEncoderDrop(WGPUCommandEncoder commandEncoder) {
  wgpuCommandEncoderDrop(commandEncoder);
}

void iree_wgpuComputePipelineDrop(WGPUComputePipeline computePipeline) {
  wgpuComputePipelineDrop(computePipeline);
}

void iree_wgpuPipelineLayoutDrop(WGPUPipelineLayout pipelineLayout) {
  wgpuPipelineLayoutDrop(pipelineLayout);
}

void iree_wgpuQuerySetDrop(WGPUQuerySet querySet) {
  wgpuQuerySetDrop(querySet);
}

void iree_wgpuShaderModuleDrop(WGPUShaderModule shaderModule) {
  wgpuShaderModuleDrop(shaderModule);
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
      *sync_status = IREEWGPUBufferMapAsyncStatus_Success;
      break;
    case WGPUBufferMapAsyncStatus_Error:
      *sync_status = IREEWGPUBufferMapAsyncStatus_Error;
      break;
    default:
    case WGPUBufferMapAsyncStatus_Unknown:
      *sync_status = IREEWGPUBufferMapAsyncStatus_Unknown;
      break;
    case WGPUBufferMapAsyncStatus_DeviceLost:
      *sync_status = IREEWGPUBufferMapAsyncStatus_DeviceLost;
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
  wgpuDevicePoll(device, /*force_wait=*/true);
  return status;
}
